from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from tgvf_rl.cli import main
from tgvf_rl.conditioning import TargetConditioningProviderKind
from tgvf_rl.objectives.base import spec_identity_sha256
from tgvf_rl.representation.training.config import (
    ACCEPTED_QWEN3_ATTENTION_BACKEND,
    ACCEPTED_QWEN3_MODEL_DTYPE,
    ACCEPTED_QWEN3_MODEL_NAME,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4,
    REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5,
    REPRESENTATION_TRAINING_SCOPE,
    RepresentationDataConfigV2,
    RepresentationObjectiveExecutionConfigV2,
    RepresentationObjectiveExecutionConfigV3,
    load_representation_training_config,
)
from tgvf_rl.representation import TGVFAdapterVariant
from tgvf_rl.representation.training.checkpoint import (
    REPRESENTATION_ACCUMULATION_SCHEMA_VERSION,
    REPRESENTATION_ACCUMULATION_SCHEMA_VERSION_V2,
    RepresentationAccumulationIdentity,
    RepresentationAccumulationIdentityV2,
)
from tgvf_rl.representation.training.data import SplitOverlapReport
from tgvf_rl.representation.training.distributed_checkpoint import (
    DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION,
)
from tgvf_rl.representation.training.losses import MatrixCEScoreMode
from tgvf_rl.representation.training.native_pipeline import (
    REPRESENTATION_PROMPT_IDENTITY,
    REPRESENTATION_PROMPT_SCHEMA_VERSION,
)
from tgvf_rl.representation.training.objective import (
    RepresentationObjectiveConfigV2,
    RepresentationObjectiveKind,
)
from tgvf_rl.representation.training.runtime import (
    ACCEPTED_QWEN3_CHAT_TEMPLATE_SHA256,
    ACCEPTED_QWEN3_MODEL_PATH,
    ACCEPTED_QWEN3_TOKENIZER_LENGTH,
    qwen3_input_embedding_identity,
)


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _write_config(
    tmp_path: Path,
    *,
    provider: str = "contextual_hidden_state",
    suffix: str = "",
    verify_data: bool = True,
) -> Path:
    train_bytes = b'{"split":"train"}\n'
    validation_bytes = b'{"split":"validation"}\n'
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_bytes(train_bytes)
    validation.write_bytes(validation_bytes)
    prompt = "{question}"
    prompt_sha256 = _sha(prompt.encode("utf-8"))
    if provider == "contextual_hidden_state":
        conditioning = (
            '[conditioning]\nprovider = "contextual_hidden_state"\nhidden_layer = -1'
        )
    elif provider == "target_token_embedding":
        embedding_identity = (
            f"{ACCEPTED_QWEN3_MODEL_PATH}::language_model.input_embeddings"
        )
        conditioning = (
            '[conditioning]\nprovider = "target_token_embedding"\n'
            f"embedding_identity = {json.dumps(embedding_identity)}"
        )
    else:
        raise AssertionError(provider)
    content = f"""
schema_version = "{REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION}"
scope = "{REPRESENTATION_TRAINING_SCOPE}"
run_id = "representation-smoke-contextual-v1"

[code]
repository = "Miocio-nora/TGVF-E2E-RL"
commit = "000275ad8803ddb3092e1c341bd496e3847d8029"
dirty = false
dirty_state_sha256 = "none"

[model]
family = "qwen3_vl"
model_name = "{ACCEPTED_QWEN3_MODEL_NAME}"
local_path = "{ACCEPTED_QWEN3_MODEL_PATH}"
tokenizer_length = {ACCEPTED_QWEN3_TOKENIZER_LENGTH}
chat_template_sha256 = "{ACCEPTED_QWEN3_CHAT_TEMPLATE_SHA256}"
dtype = "{ACCEPTED_QWEN3_MODEL_DTYPE}"
attention_backend = "{ACCEPTED_QWEN3_ATTENTION_BACKEND}"
local_files_only = true
trust_remote_code = false
tokenizer_resize = false

{conditioning}

[data]
warn_on_target_leakage = true
require_disjoint_validation = true

[data.train]
jsonl_path = "{train}"
source_sha256 = "{_sha(train_bytes) if verify_data else "1" * 64}"
batch_size = 5
sampler_seed = 71

[data.validation]
jsonl_path = "{validation}"
source_sha256 = "{_sha(validation_bytes)}"
batch_size = 5
sampler_seed = 73

[prompt]
identity = "{REPRESENTATION_PROMPT_IDENTITY}"
template = {json.dumps(prompt)}
sha256 = "{prompt_sha256}"

[objective]
identity = "matrix-ce-plus-l-gen-smoke-v1"
kind = "matrix_ce_and_l_gen"
matrix_ce_weight = 1.0
l_gen_weight = 0.25
manifold_enabled = false
manifold_weight = 0.0
norm_loss = "unset_not_implemented"

[optimizer]
type = "adamw"
learning_rate = 0.0003
betas = [0.9, 0.999]
eps = 0.00000001
weight_decay = 0.01
amsgrad = false
maximize = false
foreach = false
capturable = false
differentiable = false
fused = false
decoupled_weight_decay = true

[scheduler]
kind = "linear_warmup_decay"
total_steps = 3
warmup_steps = 1

[execution]
precision = "bf16"
max_grad_norm = 1.0
require_all_adapter_gradients = true
gradient_clip_norm_type = 2.0
gradient_clip_error_if_nonfinite = true

[initialization]
kind = "fresh_random"
seed = 20260719
source_artifact_sha256 = "none"
allow_legacy_checkpoint_initialization = false

[fsdp2]
strategy = "fsdp2"
world_size = 2
physical_gpu_ids = [2, 3]
logical_gpu_ids = [0, 1]
device_type = "cuda"
mesh_dim_name = "fsdp"
mesh_shape = [2]
reshard_after_forward = true
parameter_dtype = "bfloat16"
reduce_dtype = "float32"
output_dtype = "bfloat16"
cast_forward_inputs = true
offload_policy = "none"

[training]
gradient_accumulation_steps = 2
target_optimizer_steps = 3
validation_every_optimizer_steps = 1
log_every_optimizer_steps = 1

[output]
final_artifact_path = "{tmp_path / "output" / "adapter.pt"}"
metrics_jsonl_path = "{tmp_path / "output" / "metrics.jsonl"}"
allow_overwrite = false

[resume]
enabled = false
checkpoint_path = "none"
strict_identity = true

[checkpoint]
directory = "{tmp_path / "checkpoints"}"
filename_prefix = "representation-smoke"
save_every_optimizer_steps = 1
save_final = true
keep_last = 2
strict_identity = true
optimizer_boundary_only = true
format = "distributed-representation-checkpoint-v1"
{suffix}
""".lstrip()
    path = tmp_path / f"representation-{provider}.toml"
    path.write_text(content, encoding="utf-8")
    return path


def _upgrade_config_to_v2(path: Path) -> Path:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        f'schema_version = "{REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION}"',
        f'schema_version = "{REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2}"',
    )
    text = text.replace(
        'kind = "matrix_ce_and_l_gen"',
        'kind = "matrix_ce_l_gen_and_norm"',
    ).replace(
        'norm_loss = "unset_not_implemented"',
        "norm_weight = 0.1",
    )
    text = text.replace(
        '[scheduler]\nkind = "linear_warmup_decay"\ntotal_steps = 3\nwarmup_steps = 1',
        '[scheduler]\nkind = "historical_cosine"\n'
        "total_steps = 2000\nwarmup_steps = 100\nmin_lr_ratio = 0.1",
    )
    text = text.replace(
        "target_optimizer_steps = 3",
        "target_optimizer_steps = 2",
    )
    text = text.replace(
        "require_disjoint_validation = true",
        'split_overlap_policy = "require_disjoint"\n'
        f'expected_overlap_report_sha256 = "{SplitOverlapReport(records=()).identity_sha256}"',
    )
    text = text.replace(
        'format = "distributed-representation-checkpoint-v1"',
        'format = "distributed-representation-checkpoint-v2"',
    )
    path.write_text(text, encoding="utf-8")
    return path


def _upgrade_config_to_v3(path: Path) -> Path:
    path = _upgrade_config_to_v2(path)
    text = path.read_text(encoding="utf-8").replace(
        f'schema_version = "{REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2}"',
        f'schema_version = "{REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3}"',
    )
    text = text.replace(
        "norm_weight = 0.1",
        'norm_weight = 0.1\nmatrix_ce_mode = "balanced"',
    )
    text = text.replace(
        "[prompt]\n",
        f'[prompt]\nschema_version = "{REPRESENTATION_PROMPT_SCHEMA_VERSION}"\n',
        1,
    )
    path.write_text(text, encoding="utf-8")
    return path


def _upgrade_config_to_v4(path: Path, *, evaluation_table: str) -> Path:
    path = _upgrade_config_to_v3(path)
    text = path.read_text(encoding="utf-8").replace(
        f'schema_version = "{REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3}"',
        f'schema_version = "{REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4}"',
    )
    text = text.replace(
        "\n[output]\n",
        f"\n[post_training_internal_evaluation]\n{evaluation_table}\n\n[output]\n",
        1,
    )
    path.write_text(text, encoding="utf-8")
    return path


def _upgrade_config_to_v5(path: Path, *, variant: str) -> Path:
    path = _upgrade_config_to_v4(path, evaluation_table="enabled = false")
    text = path.read_text(encoding="utf-8").replace(
        f'schema_version = "{REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4}"',
        f'schema_version = "{REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5}"',
    )
    text = text.replace(
        "\n[conditioning]\n",
        f'\n[adapter]\nvariant = "{variant}"\n\n[conditioning]\n',
        1,
    )
    path.write_text(text, encoding="utf-8")
    return path


def test_complete_config_maps_to_runtime_contracts_and_binds_both_hashes(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path)

    config = load_representation_training_config(path)

    assert config.code_identity.repository == "Miocio-nora/TGVF-E2E-RL"
    assert config.model_identity.revision_or_path == ACCEPTED_QWEN3_MODEL_PATH
    assert (
        config.provider.provider
        is TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
    )
    assert config.prompt.expected_sha256 == config.prompt.sha256
    assert config.objective.objective.matrix_ce_weight == 1.0
    assert config.objective.objective.l_gen_weight == 0.25
    assert config.objective.manifold_weight == 0.0
    assert config.initialization.source_artifact_sha256 is None
    assert config.optimizer.torch_options["foreach"] is False
    assert config.fsdp2.runtime_config.world_size == 2
    assert config.training.groups_per_rank_per_optimizer_step == 1
    assert type(config.accumulation_identity) is RepresentationAccumulationIdentity
    assert (
        config.accumulation_identity.schema_version
        == REPRESENTATION_ACCUMULATION_SCHEMA_VERSION
    )
    assert config.accumulation_identity.gradient_accumulation_steps == 2
    assert (
        config.checkpoint.format == DISTRIBUTED_REPRESENTATION_CHECKPOINT_SCHEMA_VERSION
    )
    assert config.source_toml_sha256 == _sha(path.read_bytes())
    assert len(config.canonical_config_sha256) == 64


def test_world4_ga2_config_preserves_eight_global_matrices(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path)
    text = path.read_text(encoding="utf-8")
    text = text.replace("world_size = 2", "world_size = 4", 1)
    text = text.replace(
        "physical_gpu_ids = [2, 3]",
        "physical_gpu_ids = [0, 1, 2, 3]",
        1,
    )
    text = text.replace(
        "logical_gpu_ids = [0, 1]",
        "logical_gpu_ids = [0, 1, 2, 3]",
        1,
    )
    text = text.replace("mesh_shape = [2]", "mesh_shape = [4]", 1)
    path.write_text(text, encoding="utf-8")

    config = load_representation_training_config(path)

    assert config.fsdp2.world_size == 4
    assert config.fsdp2.physical_gpu_ids == (0, 1, 2, 3)
    assert config.fsdp2.logical_gpu_ids == (0, 1, 2, 3)
    assert config.fsdp2.mesh_shape == (4,)
    assert config.training.gradient_accumulation_steps == 2
    assert config.accumulation_identity.data_parallel_world_size == 4
    assert config.training.gradient_accumulation_steps * config.fsdp2.world_size == 8


def test_representation_topology_rejects_unsupported_world3(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path)
    text = path.read_text(encoding="utf-8")
    text = text.replace("world_size = 2", "world_size = 3", 1)
    text = text.replace("physical_gpu_ids = [2, 3]", "physical_gpu_ids = [0, 1, 2]", 1)
    text = text.replace("logical_gpu_ids = [0, 1]", "logical_gpu_ids = [0, 1, 2]", 1)
    text = text.replace("mesh_shape = [2]", "mesh_shape = [3]", 1)
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="world_size must be 2 or 4"):
        load_representation_training_config(path, verify_external_files=False)


def test_model_image_max_pixels_is_optional_and_exposed_by_validation(
    tmp_path: Path,
) -> None:
    uncapped = load_representation_training_config(
        _write_config(tmp_path),
        verify_external_files=False,
    )
    uncapped_model_payload = uncapped.validation_payload()["model"]
    assert uncapped.model.image_max_pixels is None
    assert isinstance(uncapped_model_payload, dict)
    assert uncapped_model_payload["image_max_pixels"] is None

    path = _write_config(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "tokenizer_resize = false",
            "tokenizer_resize = false\nimage_max_pixels = 262144",
        ),
        encoding="utf-8",
    )

    capped = load_representation_training_config(path, verify_external_files=False)
    capped_model_payload = capped.validation_payload()["model"]

    assert capped.model.image_max_pixels == 262144
    assert isinstance(capped_model_payload, dict)
    assert capped_model_payload["image_max_pixels"] == 262144


def test_direct_groups_select_versioned_accumulation_identity_and_partition_evenly(
    tmp_path: Path,
) -> None:
    path = _upgrade_config_to_v2(_write_config(tmp_path))
    text = path.read_text(encoding="utf-8").replace(
        "gradient_accumulation_steps = 2",
        "gradient_accumulation_steps = 2\ngroups_per_rank_per_optimizer_step = 4",
    )
    path.write_text(text, encoding="utf-8")

    config = load_representation_training_config(path)
    accumulation = config.accumulation_identity

    assert config.training.groups_per_rank_per_optimizer_step == 4
    assert type(accumulation) is RepresentationAccumulationIdentityV2
    assert accumulation.groups_per_rank_per_optimizer_step == 4
    assert accumulation.gradient_accumulation_steps == 2
    assert accumulation.groups_per_accumulation_microstep == 2
    assert accumulation.schema_version == REPRESENTATION_ACCUMULATION_SCHEMA_VERSION_V2

    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "groups_per_rank_per_optimizer_step = 4",
            "groups_per_rank_per_optimizer_step = 3",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="evenly divisible"):
        load_representation_training_config(path, verify_external_files=False)

    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "groups_per_rank_per_optimizer_step = 3",
            "groups_per_rank_per_optimizer_step = 2",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="more than one group per accumulation"):
        load_representation_training_config(path, verify_external_files=False)


@pytest.mark.parametrize("groups", [0, -1])
def test_direct_groups_must_be_positive(tmp_path: Path, groups: int) -> None:
    path = _write_config(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "gradient_accumulation_steps = 2",
            "gradient_accumulation_steps = 1\n"
            f"groups_per_rank_per_optimizer_step = {groups}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="groups_per_rank_per_optimizer_step"):
        load_representation_training_config(path, verify_external_files=False)


def test_v2_binds_norm_and_2000_step_cosine_while_allowing_bounded_target(
    tmp_path: Path,
) -> None:
    path = _upgrade_config_to_v2(_write_config(tmp_path))

    config = load_representation_training_config(path)

    assert config.schema_version == REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V2
    assert isinstance(config.data, RepresentationDataConfigV2)
    assert config.data.expected_overlap_report_sha256 == (
        SplitOverlapReport(records=()).identity_sha256
    )
    assert isinstance(config.objective, RepresentationObjectiveExecutionConfigV2)
    assert config.objective.objective.norm_weight == 0.1
    assert config.objective.manifold_weight == 0.0
    assert config.scheduler.total_steps == 2000
    assert config.scheduler.warmup_steps == 100
    assert config.scheduler.min_lr_ratio == 0.1
    assert config.training.target_optimizer_steps == 2

    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "target_optimizer_steps = 2",
            "target_optimizer_steps = 2001",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot exceed.*scheduler horizon"):
        load_representation_training_config(path, verify_external_files=False)


def test_v2_has_no_norm_mode_and_v1_remains_loadable_unchanged(tmp_path: Path) -> None:
    v1_path = _write_config(tmp_path)
    v1_raw = v1_path.read_bytes()
    v1 = load_representation_training_config(v1_path, verify_external_files=False)
    assert v1.schema_version == REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION
    assert v1_path.read_bytes() == v1_raw

    v2_path = _upgrade_config_to_v2(v1_path)
    v2_path.write_text(
        v2_path.read_text(encoding="utf-8").replace(
            "norm_weight = 0.1",
            'norm_weight = 0.1\nnorm_mode = "another-target"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown.*norm_mode"):
        load_representation_training_config(v2_path, verify_external_files=False)


def test_v3_selects_balanced_matrix_ce_and_defaults_temperature(tmp_path: Path) -> None:
    path = _upgrade_config_to_v3(_write_config(tmp_path))
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'matrix_ce_mode = "balanced"\n',
            "",
        ),
        encoding="utf-8",
    )

    defaulted = load_representation_training_config(
        path,
        verify_external_files=False,
    )

    assert defaulted.schema_version == REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V3
    assert defaulted.prompt.schema_version == REPRESENTATION_PROMPT_SCHEMA_VERSION
    assert defaulted.prompt.template == "{question}"
    assert isinstance(defaulted.objective, RepresentationObjectiveExecutionConfigV3)
    assert defaulted.objective.objective.matrix_ce_mode is MatrixCEScoreMode.BALANCED
    assert defaulted.objective.objective.matrix_ce_temperature == 1.0
    assert defaulted.validation_payload()["matrix_ce_mode"] == "balanced"
    assert defaulted.validation_payload()["matrix_ce_temperature"] == 1.0
    assert (
        defaulted.validation_payload()["prompt_schema_version"]
        == REPRESENTATION_PROMPT_SCHEMA_VERSION
    )

    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "norm_weight = 0.1",
            'norm_weight = 0.1\nmatrix_ce_mode = "balanced"',
        ),
        encoding="utf-8",
    )
    explicit_balanced = load_representation_training_config(
        path,
        verify_external_files=False,
    )
    assert spec_identity_sha256(explicit_balanced.objective.objective) == (
        spec_identity_sha256(defaulted.objective.objective)
    )
    assert explicit_balanced.canonical_config_sha256 != (
        defaulted.canonical_config_sha256
    )

    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'matrix_ce_mode = "balanced"',
            'matrix_ce_mode = "balanced"\nmatrix_ce_temperature = 0.5',
        ),
        encoding="utf-8",
    )
    explicit = load_representation_training_config(
        path,
        verify_external_files=False,
    )
    assert explicit.objective.objective.matrix_ce_temperature == 0.5
    assert spec_identity_sha256(explicit.objective.objective) != (
        spec_identity_sha256(defaulted.objective.objective)
    )


def test_v3_legacy_mode_is_selectable_but_cannot_be_tempered(tmp_path: Path) -> None:
    path = _upgrade_config_to_v3(_write_config(tmp_path))
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'matrix_ce_mode = "balanced"',
            'matrix_ce_mode = "legacy_summed_nll"',
        ),
        encoding="utf-8",
    )
    legacy = load_representation_training_config(
        path,
        verify_external_files=False,
    )
    assert legacy.objective.objective.matrix_ce_mode is (
        MatrixCEScoreMode.LEGACY_SUMMED_NLL
    )
    assert legacy.objective.objective.matrix_ce_temperature == 1.0

    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'matrix_ce_mode = "legacy_summed_nll"',
            'matrix_ce_mode = "legacy_summed_nll"\nmatrix_ce_temperature = 0.5',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="legacy_summed_nll requires"):
        load_representation_training_config(path, verify_external_files=False)


def test_v4_has_an_explicit_disabled_post_training_evaluation_switch(
    tmp_path: Path,
) -> None:
    path = _upgrade_config_to_v4(
        _write_config(tmp_path),
        evaluation_table="enabled = false",
    )

    config = load_representation_training_config(path, verify_external_files=False)

    assert config.schema_version == REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V4
    assert config.post_training_internal_evaluation is not None
    assert config.post_training_internal_evaluation.enabled is False
    assert (
        config.validation_payload()["post_training_internal_evaluation_enabled"]
        is False
    )


def test_v5_content_binds_main_d_only_adapter_variant(tmp_path: Path) -> None:
    path = _upgrade_config_to_v5(_write_config(tmp_path), variant="main_d_only")

    config = load_representation_training_config(path, verify_external_files=False)

    assert config.schema_version == REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5
    assert config.adapter_variant is TGVFAdapterVariant.MAIN_D_ONLY
    assert config.validation_payload()["adapter_variant"] == "main_d_only"


def test_v5_content_binds_vision_routing_adapter_variant(tmp_path: Path) -> None:
    path = _upgrade_config_to_v5(
        _write_config(tmp_path),
        variant="full_d_deepstack_vision_routing",
    )

    config = load_representation_training_config(path, verify_external_files=False)

    assert config.schema_version == REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5
    assert config.adapter_variant is (
        TGVFAdapterVariant.FULL_D_DEEPSTACK_VISION_ROUTING
    )
    assert (
        config.validation_payload()["adapter_variant"]
        == "full_d_deepstack_vision_routing"
    )


def test_v5_content_binds_visual_barycentric_adapter_variant(tmp_path: Path) -> None:
    path = _upgrade_config_to_v5(
        _write_config(tmp_path),
        variant="full_d_deepstack_visual_barycentric",
    )

    config = load_representation_training_config(path, verify_external_files=False)

    assert config.schema_version == REPRESENTATION_TRAINING_CONFIG_SCHEMA_VERSION_V5
    assert config.adapter_variant is (
        TGVFAdapterVariant.FULL_D_DEEPSTACK_VISUAL_BARYCENTRIC
    )
    assert (
        config.validation_payload()["adapter_variant"]
        == "full_d_deepstack_visual_barycentric"
    )


def test_v4_enabled_post_training_evaluation_requires_every_identity(
    tmp_path: Path,
) -> None:
    groups = tmp_path / "groups.json"
    counterfactuals = tmp_path / "counterfactuals.json"
    groups.write_text("{}", encoding="utf-8")
    counterfactuals.write_text("{}", encoding="utf-8")
    table = "\n".join(
        (
            "enabled = true",
            'evaluation_id = "representation-internal-eval-v1"',
            f'ordered_group_manifest_path = "{groups}"',
            f'ordered_group_manifest_sha256 = "{_sha(groups.read_bytes())}"',
            f'counterfactual_manifest_path = "{counterfactuals}"',
            f'counterfactual_manifest_sha256 = "{_sha(counterfactuals.read_bytes())}"',
            f'report_path = "{tmp_path / "report.json"}"',
            "random_seed = 20260525",
            "max_new_tokens = 64",
            "eos_token_ids = [151645]",
        )
    )
    path = _upgrade_config_to_v4(_write_config(tmp_path), evaluation_table=table)

    config = load_representation_training_config(path, verify_external_files=False)

    evaluation = config.post_training_internal_evaluation
    assert evaluation is not None and evaluation.enabled
    assert evaluation.ordered_group_manifest_path == groups
    assert evaluation.counterfactual_manifest_path == counterfactuals
    assert evaluation.eos_token_ids == (151645,)

    path.write_text(
        path.read_text(encoding="utf-8").replace("max_new_tokens = 64\n", ""),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing=.*max_new_tokens"):
        load_representation_training_config(path, verify_external_files=False)


def test_v3_rejects_post_training_evaluation_table(tmp_path: Path) -> None:
    path = _upgrade_config_to_v3(_write_config(tmp_path))
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "\n[output]\n",
            "\n[post_training_internal_evaluation]\nenabled = false\n\n[output]\n",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown=.*post_training_internal_evaluation"):
        load_representation_training_config(path, verify_external_files=False)


def test_existing_v2_objective_identity_remains_unchanged() -> None:
    objective = RepresentationObjectiveConfigV2(
        identity="matrix-ce-l-gen-norm-rpi-20260719-smoke-v1",
        kind=RepresentationObjectiveKind.MATRIX_CE_L_GEN_AND_NORM,
        matrix_ce_weight=1.0,
        l_gen_weight=1.0,
        norm_weight=0.1,
    )

    assert spec_identity_sha256(objective) == (
        "3203b12dc9474f60e8fc0b1a224471fedfb648ee3dd1fa73dcd79a08a487d7c9"
    )


def test_v3_rejects_unaccepted_or_target_bearing_prompt_contract(
    tmp_path: Path,
) -> None:
    path = _upgrade_config_to_v3(_write_config(tmp_path))
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            f'schema_version = "{REPRESENTATION_PROMPT_SCHEMA_VERSION}"',
            'schema_version = "unaccepted_representation_prompt"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prompt schema mismatch"):
        load_representation_training_config(path, verify_external_files=False)

    path = _upgrade_config_to_v3(_write_config(tmp_path))
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            REPRESENTATION_PROMPT_IDENTITY,
            "unbound-question-only-alias",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fixed image-question prompt identity"):
        load_representation_training_config(path, verify_external_files=False)

    path = _upgrade_config_to_v3(_write_config(tmp_path))
    rejected = "Question: {question}\\nRequested target: {target}"
    path.write_text(
        path.read_text(encoding="utf-8")
        .replace(
            f"template = {json.dumps('{question}')}",
            f"template = {json.dumps(rejected)}",
        )
        .replace(
            f'sha256 = "{_sha(b"{question}")}"',
            f'sha256 = "{_sha(rejected.encode("utf-8"))}"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires template exactly"):
        load_representation_training_config(path, verify_external_files=False)


def test_target_token_embedding_is_a_real_exclusive_provider_choice(
    tmp_path: Path,
) -> None:
    config = load_representation_training_config(
        _write_config(tmp_path, provider="target_token_embedding")
    )

    assert (
        config.provider.provider
        is TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING
    )
    assert config.provider.embedding_identity == qwen3_input_embedding_identity(
        config.model_identity
    )
    assert config.provider.hidden_layer is None


def test_unknown_and_missing_fields_fail_closed(tmp_path: Path) -> None:
    unknown = _write_config(tmp_path, suffix="unknown_switch = true")
    with pytest.raises(ValueError, match="unknown"):
        load_representation_training_config(unknown, verify_external_files=False)

    missing = _write_config(tmp_path)
    text = missing.read_text(encoding="utf-8").replace(
        "require_all_adapter_gradients = true\n", ""
    )
    missing.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_representation_training_config(missing, verify_external_files=False)


def test_every_checkpoint_step_must_have_a_durable_train_metric(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path)
    base = path.read_text(encoding="utf-8").replace(
        "log_every_optimizer_steps = 1",
        "log_every_optimizer_steps = 3",
    )
    variants = (
        base.replace(
            "save_every_optimizer_steps = 1",
            "save_every_optimizer_steps = 2",
        ),
        base.replace(
            "save_every_optimizer_steps = 1",
            "save_every_optimizer_steps = 3",
        )
        .replace(
            "target_optimizer_steps = 3",
            "target_optimizer_steps = 4",
        )
        .replace(
            "total_steps = 3",
            "total_steps = 4",
        ),
    )
    for text in variants:
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="durable train metric"):
            load_representation_training_config(path, verify_external_files=False)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "physical_gpu_ids = [2, 3]",
            "physical_gpu_ids = [2, 2]",
            "distinct non-negative",
        ),
        (
            "manifold_enabled = false",
            "manifold_enabled = true",
            "manifold",
        ),
        (
            'kind = "fresh_random"',
            'kind = "adapter_artifact"',
            "fresh_random",
        ),
        (
            "allow_legacy_checkpoint_initialization = false",
            "allow_legacy_checkpoint_initialization = true",
            "legacy",
        ),
        (
            "l_gen_weight = 0.25",
            "l_gen_weight = 0.0",
            "nonzero L_gen",
        ),
        (
            'attention_backend = "sdpa"',
            'attention_backend = "flash_attention_2"',
            "SDPA",
        ),
    ],
)
def test_fixed_safety_and_scientific_boundaries_fail_closed(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = _write_config(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        load_representation_training_config(path, verify_external_files=False)


def test_jsonl_bytes_are_verified_but_validation_never_launches_gpu(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path, verify_data=False)

    with pytest.raises(ValueError, match="data.train.source_sha256 mismatch"):
        load_representation_training_config(path)


def test_enabled_resume_requires_a_distributed_checkpoint_directory(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path)
    resume_path = tmp_path / "resume-checkpoint"
    resume_path.mkdir()
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "[resume]\nenabled = false", "[resume]\nenabled = true"
    ).replace('checkpoint_path = "none"', f'checkpoint_path = "{resume_path}"')
    path.write_text(text, encoding="utf-8")

    config = load_representation_training_config(path)
    assert config.resume.checkpoint_path == resume_path

    resume_path.rmdir()
    resume_path.write_bytes(b"plain checkpoint files are rejected")
    with pytest.raises(ValueError, match="distributed checkpoint directory"):
        load_representation_training_config(path)


def test_canonical_digest_ignores_toml_formatting_but_raw_digest_does_not(
    tmp_path: Path,
) -> None:
    first = _write_config(tmp_path)
    first_config = load_representation_training_config(
        first, verify_external_files=False
    )
    second = tmp_path / "formatted.toml"
    second.write_text(
        first.read_text(encoding="utf-8") + "\n# identity-preserving comment\n",
        encoding="utf-8",
    )
    second_config = load_representation_training_config(
        second, verify_external_files=False
    )

    assert first_config.canonical_config_sha256 == second_config.canonical_config_sha256
    assert first_config.source_toml_sha256 != second_config.source_toml_sha256


def test_validation_cli_emits_identity_without_starting_training(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_config(tmp_path)

    assert main(["validate-representation-config", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["canonical_config_sha256"]
    assert payload["conditioning_provider"] == "contextual_hidden_state"
    assert payload["physical_gpu_ids"] == [2, 3]
    assert payload["gpu_work_launched"] is False
