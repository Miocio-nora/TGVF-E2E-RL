from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tgvf_rl.conditioning import TargetConditioningProviderKind
from tgvf_rl.data import (
    DEEPEYES47K_DATASET_ID,
    DEEPEYES47K_SAMPLES_FILE,
    DEEPEYES47K_SCHEMA_VERSION,
    DEEPEYES47K_SHUFFLE_ALGORITHM,
    DEEPEYES47K_SNAPSHOT,
    DEEPEYES47K_TOTAL_ROWS,
    DeepEyes47KRuntimeBinding,
    POLICY_T1_ARXIVQA_DATASET_KIND,
    PolicyT1DecisionStage,
    PolicyT1RLRuntimeBinding,
    SelectionCandidate,
    canonical_json_line,
    materialize_policy_t1_arxivqa_rl_dataset,
)
from tgvf_rl.policy.config import (
    POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256,
    POLICY_PILOT_V1_MODEL_FAMILY,
    POLICY_PILOT_V1_MODEL_NAME,
    POLICY_PILOT_V1_MODEL_PATH,
    POLICY_PILOT_V1_TOKENIZER_LENGTH,
    POLICY_PILOT_V1_VLLM_VERSION,
    PolicyVisualToolExperimentConfig,
)
from tgvf_rl.policy.run_config import (
    POLICY_E2E_AGENT_LOOP_CONFIG_PATH,
    POLICY_E2E_FORMAL_PILOT_CONFIG_SCHEMA,
    POLICY_E2E_RUNTIME_INVOCATION_FACTORY_FQN,
    POLICY_E2E_SMOKE_ANSWER_VERIFIER_SHA256,
    POLICY_E2E_SMOKE_CAP_ERROR_SHA256,
    POLICY_E2E_SMOKE_CONFIG_SCHEMA,
    POLICY_E2E_SMOKE_SEED_DERIVATION_NAME,
    POLICY_E2E_SMOKE_SEED_DERIVATION_SHA256,
    POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256,
    POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA,
    formal_deepeyes47k_iteration_identity_sha256,
    load_policy_e2e_smoke_run_config,
)
from tgvf_rl.framework.verl.launcher import (
    DEEPEYES47K_DATASET_CLASS_NAME,
    DEEPEYES47K_DATASET_MODULE_PATH,
    POLICY_T1_ARXIVQA_DATASET_CLASS_NAME,
    POLICY_T1_ARXIVQA_DATASET_MODULE_PATH,
    NATIVE_INVOCATION_FACTORY_FQN,
    POLICY_CHECKPOINT_ENGINE_MANAGER_FQN,
    SELECTED_SAMPLE_DATASET_MODULE_PATH,
    UPSTREAM_VERL_V0_RUNNER_FQN,
    build_policy_e2e_smoke_verl_plan,
    compose_upstream_verl_config,
)
from tgvf_rl.framework.verl.smoke_dataset import (
    TGVFSelectedSampleDataset,
    VerlSelectedSampleDatasetBinding,
)
from tgvf_rl.policy.launch import policy_child_environment
from tgvf_rl.protocol import (
    IMAGE_ZOOM_IN_TOOL_SCHEMA_SHA256,
    TGVF_FOCUS_TOOL_SCHEMA_SHA256,
    NativeToolCapabilityProfile,
    native_assistant_dialect_for_model,
    visual_tool_prompt_identity,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
COMMIT = "c" * 40
SAMPLE_ID = "deepeyes47k:fixture-selected-sample"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _prepare_external_inputs(root: Path) -> dict[str, object]:
    dataset_root = root / "deepeyes"
    dataset_root.mkdir()
    images_root = dataset_root / "images"
    images_root.mkdir()
    image_bytes = b"selected image bytes"
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    image_path = images_root / f"{image_sha256}.png"
    image_path.write_bytes(image_bytes)
    selected_sample = {
        "sample_id": SAMPLE_ID,
        "image": {
            "path": f"images/{image_path.name}",
            "sha256": image_sha256,
        },
        "extra_info": {"question": "Which option is correct?\nA. first\nB. second"},
        "reward_model": {"ground_truth": "B"},
        "data_source": "multiple_choice",
        "task_kind": "mcq",
        "provenance": {
            "dataset_id": DEEPEYES47K_DATASET_ID,
            "snapshot": DEEPEYES47K_SNAPSHOT,
            "source_file": "fixture.parquet",
            "source_file_sha256": SHA_A,
            "source_row_index": 0,
        },
    }
    samples_bytes = _canonical(selected_sample) + b"\n"
    (dataset_root / DEEPEYES47K_SAMPLES_FILE).write_bytes(samples_bytes)
    samples_sha256 = hashlib.sha256(samples_bytes).hexdigest()
    manifest = {
        "schema_version": DEEPEYES47K_SCHEMA_VERSION,
        "dataset_id": DEEPEYES47K_DATASET_ID,
        "snapshot": DEEPEYES47K_SNAPSHOT,
        "fixture": False,
        "source_files": [],
        "source_total_rows": DEEPEYES47K_TOTAL_ROWS,
        "sample_count": DEEPEYES47K_TOTAL_ROWS,
        "shuffle": {"algorithm": DEEPEYES47K_SHUFFLE_ALGORITHM, "seed": 42},
        "samples": {
            "path": DEEPEYES47K_SAMPLES_FILE,
            "rows": DEEPEYES47K_TOTAL_ROWS,
            "sha256": samples_sha256,
        },
        "images": {"directory": "images", "address": "sha256-of-original-bytes"},
    }
    content_sha256 = hashlib.sha256(_canonical(manifest)).hexdigest()
    manifest["content_sha256"] = content_sha256
    manifest_bytes = _canonical(manifest) + b"\n"
    (dataset_root / "manifest.json").write_bytes(manifest_bytes)
    manifest_file_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    artifact_path = root / "representation-adapter.pt"
    artifact_path.write_bytes(b"opaque representation artifact fixture")
    artifact_file_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    binding = DeepEyes47KRuntimeBinding.formal(
        manifest_file_sha256=manifest_file_sha256,
        content_sha256=content_sha256,
        shuffle_seed=42,
    )
    iteration_sha256 = formal_deepeyes47k_iteration_identity_sha256(
        binding, samples_sha256=samples_sha256
    )
    return {
        "dataset_root": dataset_root,
        "manifest_file_sha256": manifest_file_sha256,
        "content_sha256": content_sha256,
        "samples_sha256": samples_sha256,
        "iteration_sha256": iteration_sha256,
        "artifact_path": artifact_path,
        "artifact_file_sha256": artifact_file_sha256,
    }


def _q(value: object) -> str:
    return json.dumps(str(value))


def _config_text(root: Path, external: dict[str, object]) -> str:
    output_root = root / "never-created-output"
    agent_loop_sha256 = hashlib.sha256(
        POLICY_E2E_AGENT_LOOP_CONFIG_PATH.read_bytes()
    ).hexdigest()
    return f'''schema_version = "{POLICY_E2E_SMOKE_CONFIG_SCHEMA}"
formal_pilot = false
run_id = "policy-smoke-test-001"

[code]
repository = "Miocio-nora/TGVF-E2E-RL"
commit = "{COMMIT}"
dirty = false

[model]
family = "{POLICY_PILOT_V1_MODEL_FAMILY}"
name = "{POLICY_PILOT_V1_MODEL_NAME}"
path = "{POLICY_PILOT_V1_MODEL_PATH}"
tokenizer_length = {POLICY_PILOT_V1_TOKENIZER_LENGTH}
chat_template_sha256 = "{POLICY_PILOT_V1_CHAT_TEMPLATE_SHA256}"
native_deepstack_enabled = true
image_max_pixels = 262144

[dataset]
root = {_q(external["dataset_root"])}
dataset_id = "{DEEPEYES47K_DATASET_ID}"
snapshot = "{DEEPEYES47K_SNAPSHOT}"
sample_count = {DEEPEYES47K_TOTAL_ROWS}
manifest_file_sha256 = "{external["manifest_file_sha256"]}"
content_sha256 = "{external["content_sha256"]}"
samples_sha256 = "{external["samples_sha256"]}"
iteration_identity_sha256 = "{external["iteration_sha256"]}"
shuffle_seed = 42
sample_id = "{SAMPLE_ID}"
cursor = 0

[representation]
artifact_path = {_q(external["artifact_path"])}
artifact_file_sha256 = "{external["artifact_file_sha256"]}"
artifact_manifest_sha256 = "{SHA_A}"
artifact_namespace = "tgvf-representation"
artifact_name = "contextual-adapter"
artifact_version = "smoke-fixture-v1"
expected_run_id = "representation-run-001"
expected_run_identity_sha256 = "{SHA_B}"

[representation.conditioning]
provider = "contextual_hidden_state"
hidden_layer = -1

[protocol]
prompt_sha256 = "{SHA_A}"
cap_error_sha256 = "{POLICY_E2E_SMOKE_CAP_ERROR_SHA256}"
tool_profile = "tgvf_only"
tool_schema_sha256 = "{TGVF_FOCUS_TOOL_SCHEMA_SHA256}"
enabled_tool_names = ["tgvf_focus_tool"]
maximum_tool_calls = 4

[sampling]
trajectories_per_prompt = 8
temperature = 1.0
top_p = 1.0
top_k = -1
min_p = 0.0
repetition_penalty = 1.0
presence_penalty = 0.0
frequency_penalty = 0.0
max_response_length = 8192
asynchronous_staleness_steps = 0
do_sample = true
backend = "vllm"
backend_version = "{POLICY_PILOT_V1_VLLM_VERSION}"
logit_processors = []
logprob_measurement = "after_sampling_transforms"
stop_token_ids = [151645]
stop_strings = ["</tool_call>"]
include_stop_str_in_output = true
ignore_eos = false
rollout_master_seed = 42
seed_derivation_name = "{POLICY_E2E_SMOKE_SEED_DERIVATION_NAME}"
seed_derivation_sha256 = "{POLICY_E2E_SMOKE_SEED_DERIVATION_SHA256}"

[reward]
task_kind = "multiple_choice"
answer_verifier = "exact_match"
answer_verifier_sha256 = "{POLICY_E2E_SMOKE_ANSWER_VERIFIER_SHA256}"
judge_mode = "not_applicable"
judge_reason = "bounded non-formal MCQ smoke"
answer_weight = 0.8
format_weight = 0.2
conditional_tool_weight = 1.2

[optimizer]
name = "adamw"
learning_rate = 0.00001
beta1 = 0.9
beta2 = 0.999
epsilon = 0.00000001
weight_decay = 0.01
maximum_gradient_norm = 1.0

[scheduler]
name = "cosine"
warmup_steps = 1
total_steps = 2
minimum_learning_rate_ratio = 0.1

[precision]
parameter_dtype = "bfloat16"
reduce_dtype = "float32"
optimizer_state_dtype = "float32"
autocast_dtype = "bfloat16"
gradient_scaler_enabled = false
allow_tf32 = true

[accumulation]
global_prompt_batch_size = 4
prompt_micro_batch_size_per_rank = 1
rollout_prompt_micro_batch_size_per_engine = 1
gradient_accumulation_steps = 1

[distributed]
physical_gpu_ids = [0, 1, 2, 3]
logical_gpu_ids = [0, 1, 2, 3]
world_size = 4
actor_logical_gpu_ids = [0, 1, 2, 3]
rollout_logical_gpu_ids = [0, 1, 2, 3]
fsdp_strategy = "fsdp2"
fsdp_reshard_after_forward = false
rollout_backend = "vllm"
vllm_tensor_parallel_size = 1
placement = "colocated"
weight_sync_mode = "nccl_lora_state_v1"
weight_sync_interval_optimizer_steps = 1

[capacity]
max_prompt_length = 4096
actor_ppo_max_token_len_per_gpu = 98304
rollout_log_prob_max_token_len_per_gpu = 98304
reference_log_prob_max_token_len_per_gpu = 98304
vllm_gpu_memory_utilization = 0.5
vllm_max_num_batched_tokens = 32768
vllm_max_model_len = 32768
vllm_max_num_seqs = 8
vllm_enable_chunked_prefill = true
vllm_enforce_eager = false

[framework]
agent_loop_config_path = {_q(POLICY_E2E_AGENT_LOOP_CONFIG_PATH)}
agent_loop_config_sha256 = "{agent_loop_sha256}"
runtime_invocation_factory_fqn = "{POLICY_E2E_RUNTIME_INVOCATION_FACTORY_FQN}"
server_timeout_seconds = 600.0

[training]
total_training_epochs = 1
maximum_optimizer_steps = 2
checkpoint_steps = [0, 1, 2]
logger = ["console"]
project_name = "tgvf-policy-rl"
validation_before_training = false
validation_frequency = -1
resume_mode = "disable"
resume_from_path = ""
maximum_actor_checkpoints_to_keep = 2

[output]
root = {_q(output_root)}
checkpoint_directory = {_q(output_root / "checkpoints")}
metrics_path = {_q(output_root / "metrics.jsonl")}
'''


def _write_config(tmp_path: Path) -> tuple[Path, str, dict[str, object]]:
    external = _prepare_external_inputs(tmp_path)
    text = _config_text(tmp_path, external)
    path = tmp_path / "policy-smoke.toml"
    path.write_text(text, encoding="utf-8")
    return path, text, external


def test_loads_complete_nonformal_smoke_and_has_stable_digest(tmp_path: Path) -> None:
    path, text, external = _write_config(tmp_path)
    output_root = tmp_path / "never-created-output"
    assert not output_root.exists()

    config = load_policy_e2e_smoke_run_config(path)
    repeated = load_policy_e2e_smoke_run_config(path)

    assert config.formal_pilot is False
    assert config.run_id == "policy-smoke-test-001"
    assert config.policy.sampling.is_run_bound
    assert config.dataset.runtime_binding.fixture is False
    assert config.dataset.runtime_binding.shuffle_seed == 42
    assert config.dataset.samples_sha256 == external["samples_sha256"]
    assert config.dataset.sample_id == SAMPLE_ID and config.dataset.cursor == 0
    assert config.representation.conditioning.provider is (
        TargetConditioningProviderKind.CONTEXTUAL_HIDDEN_STATE
    )
    assert config.reward.judge_mode == "not_applicable"
    assert config.protocol.tool_profile.value == "tgvf_only"
    assert config.distributed.physical_gpu_ids == (0, 1, 2, 3)
    assert config.distributed.logical_gpu_ids == (0, 1, 2, 3)
    assert config.distributed.fsdp_strategy == "fsdp2"
    assert config.distributed.vllm_tensor_parallel_size == 1
    assert config.capacity.vllm_max_model_len == 32768
    assert config.capacity.response_transport_length == 28672
    assert config.framework.agent_loop_config_path == POLICY_E2E_AGENT_LOOP_CONFIG_PATH
    assert config.training.checkpoint_steps == (0, 1, 2)
    assert config.as_record()["sampling"]["rollout_master_seed"] == 42
    assert config.identity_sha256 == repeated.identity_sha256
    assert (
        config.identity_sha256
        == hashlib.sha256(config.canonical_json.encode("utf-8")).hexdigest()
    )
    assert config.source_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert not output_root.exists()


def test_mixed_run_selects_full_dataset_and_real_judge_binding(tmp_path: Path) -> None:
    path, text, _ = _write_config(tmp_path)
    judge_path = (
        Path(__file__).parents[2]
        / "configs/policy/judges/qwen25_72b_rl_answer_judge_v1.json"
    ).resolve()
    judge_sha = hashlib.sha256(judge_path.read_bytes()).hexdigest()
    prompt_sha = visual_tool_prompt_identity(
        NativeToolCapabilityProfile.TGVF_ONLY,
        assistant_dialect=native_assistant_dialect_for_model(
            POLICY_PILOT_V1_MODEL_NAME
        ),
    ).bundle_sha256
    text = text.replace(
        POLICY_E2E_SMOKE_CONFIG_SCHEMA, POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA
    )
    text = text.replace(f'sample_id = "{SAMPLE_ID}"\ncursor = 0\n', "")
    text = text.replace(f'prompt_sha256 = "{SHA_A}"', f'prompt_sha256 = "{prompt_sha}"')
    text = text.replace(
        'task_kind = "multiple_choice"\n'
        'answer_verifier = "exact_match"\n'
        f'answer_verifier_sha256 = "{POLICY_E2E_SMOKE_ANSWER_VERIFIER_SHA256}"\n'
        'judge_mode = "not_applicable"\n'
        'judge_reason = "bounded non-formal MCQ smoke"',
        'task_kind = "mixed"\n'
        'answer_verifier = "rule_first_qwen25_72b"\n'
        f'answer_verifier_sha256 = "{POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256}"\n'
        'judge_mode = "qwen25_72b_semantic_fallback"\n'
        'judge_reason = "mixed integration"\n'
        f"judge_config_path = {_q(judge_path)}\n"
        f'judge_config_sha256 = "{judge_sha}"',
    )
    path.write_text(text, encoding="utf-8")

    config = load_policy_e2e_smoke_run_config(path)
    plan = build_policy_e2e_smoke_verl_plan(config)

    assert config.schema_version == POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA
    assert config.dataset.selected_sample is None
    assert config.reward.judge_config_sha256 == judge_sha
    assert plan.overrides["data.custom_cls.path"] == DEEPEYES47K_DATASET_MODULE_PATH
    assert plan.overrides["data.custom_cls.name"] == DEEPEYES47K_DATASET_CLASS_NAME
    assert (
        plan.overrides["actor_rollout_ref.rollout.custom"]["reward"][
            "judge_config_sha256"
        ]
        == judge_sha
    )

    formal_judge_path = (
        Path(__file__).parents[2]
        / "configs/policy/judges/openrouter_qwen25_72b_formal_pilot_judge_v1.json"
    ).resolve()
    formal_judge_sha = hashlib.sha256(formal_judge_path.read_bytes()).hexdigest()
    formal_text = (
        text.replace(
            POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA,
            POLICY_E2E_FORMAL_PILOT_CONFIG_SCHEMA,
        )
        .replace("formal_pilot = false", "formal_pilot = true", 1)
        .replace(str(judge_path), str(formal_judge_path), 1)
        .replace(judge_sha, formal_judge_sha, 1)
    )
    path.write_text(formal_text, encoding="utf-8")
    formal = load_policy_e2e_smoke_run_config(path)
    assert formal.formal_pilot is True
    assert formal.schema_version == POLICY_E2E_FORMAL_PILOT_CONFIG_SCHEMA


def test_mixed_run_routes_typed_t1_arxivqa_artifact_to_verl(tmp_path: Path) -> None:
    external = _prepare_external_inputs(tmp_path)
    source_row = json.loads(
        (Path(external["dataset_root"]) / DEEPEYES47K_SAMPLES_FILE)
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    image_path = (
        Path(external["dataset_root"]) / source_row["image"]["path"]
    ).resolve()
    candidate = {
        "schema_version": "tgvf.policy-selection.candidate.v1",
        "sample_id": "policy-candidate:arxivqa-fixture",
        "source": "arxivqa",
        "question": "Which option is correct?\nA. first\nB. second",
        "ground_truth": "B",
        "image": {
            "path": str(image_path),
            "sha256": source_row["image"]["sha256"],
            "width": 8,
            "height": 8,
        },
        "gt_regions": [],
        "provenance": {"fixture": True},
        "selection_metadata": {"option_count": 2},
    }
    candidate_identity = SelectionCandidate.from_record(candidate).identity_sha256
    decision = {
        "schema_version": "tgvf.policy-selection.decision.v1",
        "sample_id": candidate["sample_id"],
        "candidate_sha256": candidate_identity,
        "source": "arxivqa",
        "t1": {
            "decision": "retain",
            "full_image": {
                "accuracy": 0.5,
                "complete": True,
                "correct_count": 4,
                "expected_attempts": 8,
                "missing_indices": [],
                "observed_attempts": 8,
                "scoreable_attempts": 8,
                "status_counts": {"scored": 8},
            },
            "reason": "fixture",
        },
        "t2": {
            "decision": "not_applicable_preserve_t1",
            "gt_region": None,
            "reason": "fixture",
        },
    }
    candidates_path = tmp_path / "t1-candidates.jsonl"
    decisions_path = tmp_path / "t1-decisions.jsonl"
    candidates_path.write_bytes(canonical_json_line(candidate))
    decisions_path.write_bytes(canonical_json_line(decision))
    artifact = materialize_policy_t1_arxivqa_rl_dataset(
        candidates_path,
        decisions_path,
        tmp_path / "t1-artifact",
        decision_stage=PolicyT1DecisionStage.PROVISIONAL,
    )

    text = _config_text(tmp_path, external).replace(
        POLICY_E2E_SMOKE_CONFIG_SCHEMA, POLICY_E2E_MIXED_RUN_CONFIG_SCHEMA
    )
    dataset_start = text.index("[dataset]")
    representation_start = text.index("[representation]")
    dataset_text = f'''[dataset]
kind = "{POLICY_T1_ARXIVQA_DATASET_KIND}"
root = {_q(artifact.output_root)}
decision_stage = "provisional"
sample_count = {artifact.sample_count}
manifest_file_sha256 = "{artifact.manifest_file_sha256}"
content_sha256 = "{artifact.content_sha256}"
samples_sha256 = "{artifact.samples_sha256}"
iteration_identity_sha256 = "{artifact.iteration_identity_sha256}"
shuffle_seed = 42

'''
    text = text[:dataset_start] + dataset_text + text[representation_start:]
    prompt_sha = visual_tool_prompt_identity(
        NativeToolCapabilityProfile.CROP_ONLY,
        assistant_dialect=native_assistant_dialect_for_model(
            POLICY_PILOT_V1_MODEL_NAME
        ),
    ).bundle_sha256
    text = text.replace(f'prompt_sha256 = "{SHA_A}"', f'prompt_sha256 = "{prompt_sha}"')
    text = text.replace('tool_profile = "tgvf_only"', 'tool_profile = "crop_only"')
    text = text.replace(
        f'tool_schema_sha256 = "{TGVF_FOCUS_TOOL_SCHEMA_SHA256}"',
        f'tool_schema_sha256 = "{IMAGE_ZOOM_IN_TOOL_SCHEMA_SHA256}"',
    ).replace(
        'enabled_tool_names = ["tgvf_focus_tool"]',
        'enabled_tool_names = ["image_zoom_in_tool"]',
    )
    judge_path = (
        Path(__file__).parents[2]
        / "configs/policy/judges/qwen25_72b_rl_answer_judge_v1.json"
    ).resolve()
    judge_sha = hashlib.sha256(judge_path.read_bytes()).hexdigest()
    text = text.replace(
        'task_kind = "multiple_choice"\n'
        'answer_verifier = "exact_match"\n'
        f'answer_verifier_sha256 = "{POLICY_E2E_SMOKE_ANSWER_VERIFIER_SHA256}"\n'
        'judge_mode = "not_applicable"\n'
        'judge_reason = "bounded non-formal MCQ smoke"',
        'task_kind = "mixed"\n'
        'answer_verifier = "rule_first_qwen25_72b"\n'
        f'answer_verifier_sha256 = "{POLICY_E2E_MIXED_ANSWER_VERIFIER_SHA256}"\n'
        'judge_mode = "qwen25_72b_semantic_fallback"\n'
        'judge_reason = "T1 ArxivQA crop pilot"\n'
        f"judge_config_path = {_q(judge_path)}\n"
        f'judge_config_sha256 = "{judge_sha}"',
    )
    config_path = tmp_path / "t1-policy.toml"
    config_path.write_text(text, encoding="utf-8")

    config = load_policy_e2e_smoke_run_config(config_path)
    plan = build_policy_e2e_smoke_verl_plan(config)
    assert isinstance(config.dataset.runtime_binding, PolicyT1RLRuntimeBinding)
    assert config.dataset.kind == POLICY_T1_ARXIVQA_DATASET_KIND
    assert (
        plan.overrides["data.custom_cls.path"] == POLICY_T1_ARXIVQA_DATASET_MODULE_PATH
    )
    assert (
        plan.overrides["data.custom_cls.name"] == POLICY_T1_ARXIVQA_DATASET_CLASS_NAME
    )
    assert (
        plan.overrides["data.tgvf_policy_t1_arxivqa"]["decision_stage"] == "provisional"
    )


def test_loads_separately_identified_crop_only_experiment(tmp_path: Path) -> None:
    path, text, _ = _write_config(tmp_path)
    text = text.replace('tool_profile = "tgvf_only"', 'tool_profile = "crop_only"')
    text = text.replace(
        f'tool_schema_sha256 = "{TGVF_FOCUS_TOOL_SCHEMA_SHA256}"',
        f'tool_schema_sha256 = "{IMAGE_ZOOM_IN_TOOL_SCHEMA_SHA256}"',
    )
    text = text.replace(
        'enabled_tool_names = ["tgvf_focus_tool"]',
        'enabled_tool_names = ["image_zoom_in_tool"]',
    )
    path.write_text(text, encoding="utf-8")

    config = load_policy_e2e_smoke_run_config(path)

    assert isinstance(config.policy, PolicyVisualToolExperimentConfig)
    assert config.protocol.tool_profile is NativeToolCapabilityProfile.CROP_ONLY
    assert config.protocol.enabled_tool_names == ("image_zoom_in_tool",)


def test_auto_resume_keeps_one_identity_before_and_after_output_exists(
    tmp_path: Path,
) -> None:
    path, text, _ = _write_config(tmp_path)
    path.write_text(
        text.replace('resume_mode = "disable"', 'resume_mode = "auto"', 1),
        encoding="utf-8",
    )

    first = load_policy_e2e_smoke_run_config(path)
    first.output.root.mkdir()
    resumed = load_policy_e2e_smoke_run_config(path)

    assert first.training.resume_mode == resumed.training.resume_mode == "auto"
    assert first.training.resume_from_path is resumed.training.resume_from_path is None
    assert first.identity_sha256 == resumed.identity_sha256


def test_auto_resume_rejects_explicit_resume_path(tmp_path: Path) -> None:
    path, text, _ = _write_config(tmp_path)
    resume_path = tmp_path / "checkpoint"
    resume_path.mkdir()
    path.write_text(
        text.replace('resume_mode = "disable"', 'resume_mode = "auto"', 1).replace(
            'resume_from_path = ""',
            f"resume_from_path = {_q(resume_path)}",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="empty in auto/disable mode"):
        load_policy_e2e_smoke_run_config(path)


@pytest.mark.parametrize(
    "old, new, error",
    [
        ("min_p = 0.0\n", "", r"\[sampling\] fields differ"),
        (
            "formal_pilot = false",
            "formal_pilot = true",
            "formal_pilot mode differs",
        ),
        ('judge_mode = "not_applicable"', 'judge_mode = "qwen72b"', "judge_mode"),
        (
            'tool_profile = "tgvf_only"',
            'tool_profile = "crop_tgvf"',
            "protocol.enabled_tool_names",
        ),
        ('fsdp_strategy = "fsdp2"', 'fsdp_strategy = "fsdp1"', "fsdp_strategy"),
        (
            f'cap_error_sha256 = "{POLICY_E2E_SMOKE_CAP_ERROR_SHA256}"',
            f'cap_error_sha256 = "{SHA_A}"',
            "protocol.cap_error_sha256 differs",
        ),
        (
            f'seed_derivation_sha256 = "{POLICY_E2E_SMOKE_SEED_DERIVATION_SHA256}"',
            f'seed_derivation_sha256 = "{SHA_A}"',
            "sampling.seed_derivation_sha256 differs",
        ),
        (
            f'answer_verifier_sha256 = "{POLICY_E2E_SMOKE_ANSWER_VERIFIER_SHA256}"',
            f'answer_verifier_sha256 = "{SHA_A}"',
            "reward.answer_verifier_sha256 differs",
        ),
    ],
)
def test_rejects_missing_or_mismatched_run_inputs(
    tmp_path: Path, old: str, new: str, error: str
) -> None:
    path, text, _ = _write_config(tmp_path)
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        load_policy_e2e_smoke_run_config(path)


def test_rejects_dataset_manifest_hash_drift(tmp_path: Path) -> None:
    path, text, external = _write_config(tmp_path)
    digest = str(external["manifest_file_sha256"])
    replacement = ("0" if digest[0] != "0" else "1") + digest[1:]
    wrong_binding = DeepEyes47KRuntimeBinding.formal(
        manifest_file_sha256=replacement,
        content_sha256=str(external["content_sha256"]),
        shuffle_seed=42,
    )
    wrong_iteration = formal_deepeyes47k_iteration_identity_sha256(
        wrong_binding,
        samples_sha256=str(external["samples_sha256"]),
    )
    drifted = text.replace(digest, replacement, 1).replace(
        str(external["iteration_sha256"]), wrong_iteration, 1
    )
    path.write_text(drifted, encoding="utf-8")

    with pytest.raises(ValueError, match="manifest-file SHA256 mismatch"):
        load_policy_e2e_smoke_run_config(path)


def test_rejects_unknown_fields_and_unsafe_run_id(tmp_path: Path) -> None:
    path, text, _ = _write_config(tmp_path)
    path.write_text('surprise = "no"\n' + text, encoding="utf-8")
    with pytest.raises(ValueError, match="top-level fields differ"):
        load_policy_e2e_smoke_run_config(path)

    path.write_text(
        text.replace("policy-smoke-test-001", "../unsafe"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="safe path-independent"):
        load_policy_e2e_smoke_run_config(path)


def test_rejects_omitted_policy_owned_tool_call_closer(tmp_path: Path) -> None:
    path, text, _ = _write_config(tmp_path)
    path.write_text(
        text.replace(
            "include_stop_str_in_output = true",
            "include_stop_str_in_output = false",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="complete closing tag remains policy-sampled"):
        load_policy_e2e_smoke_run_config(path)


@pytest.mark.parametrize(
    ("old", "new", "error"),
    [
        ('"task_kind":"mcq"', '"task_kind":"open"', "task_kind='mcq'"),
        (
            '"ground_truth":"B"',
            '"ground_truth":"free form"',
            "ground truth must be a choice letter",
        ),
        (
            "Which option is correct?\\nA. first\\nB. second",
            "Which option is correct?",
            "question must contain choices",
        ),
    ],
)
def test_rejects_selected_row_that_does_not_match_mcq_reward_route(
    tmp_path: Path, old: str, new: str, error: str
) -> None:
    path, text, external = _write_config(tmp_path)
    samples_path = Path(external["dataset_root"]) / DEEPEYES47K_SAMPLES_FILE
    samples_text = samples_path.read_text(encoding="utf-8")
    assert old in samples_text
    drifted_samples = samples_text.replace(old, new, 1).encode("utf-8")
    samples_path.write_bytes(drifted_samples)
    drifted_samples_sha = hashlib.sha256(drifted_samples).hexdigest()

    manifest_path = Path(external["dataset_root"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["samples"]["sha256"] = drifted_samples_sha
    descriptor = {
        key: value for key, value in manifest.items() if key != "content_sha256"
    }
    manifest["content_sha256"] = hashlib.sha256(_canonical(descriptor)).hexdigest()
    manifest_bytes = _canonical(manifest) + b"\n"
    manifest_path.write_bytes(manifest_bytes)
    manifest_file_sha = hashlib.sha256(manifest_bytes).hexdigest()
    binding = DeepEyes47KRuntimeBinding.formal(
        manifest_file_sha256=manifest_file_sha,
        content_sha256=manifest["content_sha256"],
        shuffle_seed=42,
    )
    iteration_sha = formal_deepeyes47k_iteration_identity_sha256(
        binding, samples_sha256=drifted_samples_sha
    )
    replacements = (
        (str(external["manifest_file_sha256"]), manifest_file_sha),
        (str(external["content_sha256"]), manifest["content_sha256"]),
        (str(external["samples_sha256"]), drifted_samples_sha),
        (str(external["iteration_sha256"]), iteration_sha),
    )
    for previous, current in replacements:
        text = text.replace(previous, current, 1)
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        load_policy_e2e_smoke_run_config(path)


def test_maps_strict_smoke_to_pinned_verl_v0_hydra_without_launch(
    tmp_path: Path,
) -> None:
    path, _, _ = _write_config(tmp_path)
    config = load_policy_e2e_smoke_run_config(path)
    output_root = config.output.root
    plan = build_policy_e2e_smoke_verl_plan(config)

    assert plan.runner_fqn == UPSTREAM_VERL_V0_RUNNER_FQN
    assert plan.overrides["trainer.use_v1"] is False
    assert plan.overrides["data.custom_cls.path"] == SELECTED_SAMPLE_DATASET_MODULE_PATH
    assert plan.overrides["actor_rollout_ref.rollout.n"] == 8
    assert plan.overrides["algorithm.adv_estimator"] == "grpo"
    assert plan.overrides["actor_rollout_ref.model.lora_rank"] == 64
    assert plan.overrides["actor_rollout_ref.actor.strategy"] == "fsdp2"
    assert plan.overrides["actor_rollout_ref.actor.fsdp_config.forward_only"] is False
    assert plan.overrides["actor_rollout_ref.ref.fsdp_config.forward_only"] is True
    assert (
        plan.overrides["actor_rollout_ref.model.override_config.attn_implementation"]
        == "sdpa"
    )
    assert (
        plan.overrides["actor_rollout_ref.model.enable_gradient_checkpointing"] is False
    )
    assert plan.overrides["actor_rollout_ref.model.use_remove_padding"] is False
    assert plan.overrides["actor_rollout_ref.actor.fsdp_config.model_dtype"] == "bf16"
    assert plan.overrides["actor_rollout_ref.ref.fsdp_config.model_dtype"] == "bf16"
    assert (
        plan.overrides["actor_rollout_ref.actor.fsdp_config.use_torch_compile"] is False
    )
    assert (
        plan.overrides["actor_rollout_ref.ref.fsdp_config.use_torch_compile"] is False
    )
    # e003 v0 multiplies ppo_mini_batch_size by n internally. Actor autograd
    # consumes those expanded trajectories one at a time on each rank.
    assert plan.overrides["actor_rollout_ref.actor.ppo_mini_batch_size"] == 4
    assert plan.overrides["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == 1
    assert plan.environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert plan.environment["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"
    assert plan.environment["TGVF_POLICY_RUN_CONFIG_PATH"] == str(path)
    assert plan.environment["TGVF_POLICY_RUN_ID"] == config.run_id
    assert plan.environment["TGVF_POLICY_RUN_IDENTITY"] == config.identity_sha256
    assert plan.environment["TGVF_POLICY_STATE_DIR"] == str(
        config.output.root / "runtime-policy-state"
    )
    repository_root = Path(__file__).resolve().parents[2]
    python_include_root = repository_root / ".deps/python312-dev/root/usr/include"
    assert plan.environment["CC"] == "/usr/bin/gcc"
    assert plan.environment["CXX"] == "/usr/bin/g++"
    assert plan.environment["CPATH"] == (
        f"{python_include_root}:{python_include_root / 'python3.12'}"
    )
    assert (python_include_root / "python3.12/Python.h").is_file()
    assert plan.overrides["actor_rollout_ref.rollout.max_model_len"] == 32768
    assert plan.overrides["actor_rollout_ref.rollout.max_num_seqs"] == 8
    assert plan.overrides["data.max_prompt_length"] == 4096
    assert plan.overrides["actor_rollout_ref.rollout.response_length"] == 28672
    assert plan.overrides["data.max_response_length"] == 28672
    assert config.policy.sampling.max_response_length == 8192
    assert plan.overrides[
        "actor_rollout_ref.rollout.agent.agent_loop_config_path"
    ] == str(POLICY_E2E_AGENT_LOOP_CONFIG_PATH)
    assert plan.overrides["actor_rollout_ref.rollout.agent.num_workers"] == 4
    assert (
        plan.overrides["actor_rollout_ref.rollout.checkpoint_manager_class"]
        == POLICY_CHECKPOINT_ENGINE_MANAGER_FQN
    )
    assert plan.inherited_upstream_fields == ()
    assert plan.external_components["invocation_factory"] == (
        NATIVE_INVOCATION_FACTORY_FQN
    )
    assert plan.overrides["trainer.total_training_steps"] == 2
    assert plan.overrides["trainer.save_freq"] == 1
    assert plan.overrides["trainer.default_local_dir"] == str(
        config.output.checkpoint_directory
    )
    assert plan.launch_ready is True
    assert plan.launch_blockers == ()
    plan.assert_launch_ready()
    from verl.utils.import_utils import load_extern_object

    assert (
        load_extern_object(
            plan.overrides["data.custom_cls.path"],
            plan.overrides["data.custom_cls.name"],
        )
        is TGVFSelectedSampleDataset
    )
    assert plan.overrides["actor_rollout_ref.rollout.custom"][
        "reference_diagnostic"
    ] == {
        "enabled": True,
        "coefficient": 0.0,
        "worker_route": "colocated_frozen_base_exact_replay",
        "observation_source": "rollout_materialized_exact_bundle",
    }
    assert not output_root.exists()


def test_policy_child_environment_overrides_inherited_gpu_and_identity_values(
    tmp_path: Path,
) -> None:
    path, _, _ = _write_config(tmp_path)
    config = load_policy_e2e_smoke_run_config(path)
    plan = build_policy_e2e_smoke_verl_plan(config)

    environment = policy_child_environment(
        plan,
        base={
            "CUDA_VISIBLE_DEVICES": "7",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "0",
            "TGVF_POLICY_RUN_ID": "wrong-run",
            "UNRELATED": "preserved",
        },
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert environment["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"
    assert environment["TGVF_POLICY_RUN_ID"] == config.run_id
    assert environment["TGVF_POLICY_RUN_IDENTITY"] == config.identity_sha256
    assert environment["UNRELATED"] == "preserved"
    assert not config.output.root.exists()

    upstream_config_dir = (
        Path(__file__).resolve().parents[2]
        / ".deps"
        / "verl"
        / "verl"
        / "trainer"
        / "config"
    )
    composed = compose_upstream_verl_config(plan, config_directory=upstream_config_dir)
    assert composed.trainer.use_v1 is False
    assert composed.actor_rollout_ref.rollout.n == 8
    assert composed.actor_rollout_ref.rollout.response_length == 28672
    assert composed.data.max_response_length == 28672
    assert composed.actor_rollout_ref.rollout.agent.num_workers == 4
    assert (
        composed.actor_rollout_ref.rollout.checkpoint_manager_class
        == POLICY_CHECKPOINT_ENGINE_MANAGER_FQN
    )
    assert composed.actor_rollout_ref.actor.ppo_mini_batch_size == 4
    assert composed.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert composed.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu == 8
    assert composed.actor_rollout_ref.actor.fsdp_config.forward_only is False
    assert composed.actor_rollout_ref.ref.fsdp_config.forward_only is True
    assert (
        composed.actor_rollout_ref.model.override_config.attn_implementation == "sdpa"
    )
    assert composed.actor_rollout_ref.model.enable_gradient_checkpointing is False
    assert composed.actor_rollout_ref.model.use_remove_padding is False
    assert composed.actor_rollout_ref.actor.fsdp_config.model_dtype == "bf16"
    assert composed.actor_rollout_ref.ref.fsdp_config.model_dtype == "bf16"
    assert composed.actor_rollout_ref.actor.fsdp_config.use_torch_compile is False
    assert composed.actor_rollout_ref.ref.fsdp_config.use_torch_compile is False
    assert composed.actor_rollout_ref.rollout.full_determinism is False
    assert composed.actor_rollout_ref.actor.fsdp_config.full_determinism is True
    assert composed.actor_rollout_ref.ref.fsdp_config.full_determinism is True
    assert (
        composed.actor_rollout_ref.model.target_modules
        == config.policy.lora.target_modules
    )
    assert (
        composed.actor_rollout_ref.rollout.custom.sampling.forward_state
        == "request_seeded_batch_sensitive_v1"
    )
    assert (
        composed.actor_rollout_ref.rollout.custom.sampling.vllm_batch_invariant is False
    )
    assert (
        composed.actor_rollout_ref.rollout.custom.actor_batch_contract[
            "derived_gradient_accumulation_steps"
        ]
        == 1
    )
    assert (
        composed.actor_rollout_ref.rollout.custom.actor_batch_contract[
            "derived_actor_forward_backward_microbatches"
        ]
        == 8
    )
    assert composed.data.custom_cls.name == "TGVFSelectedSampleDataset"
    assert composed.data.custom_cls.path == SELECTED_SAMPLE_DATASET_MODULE_PATH
    assert composed.actor_rollout_ref.rollout.custom.sampling.stop_strings == [
        "</tool_call>"
    ]
    assert not config.output.root.exists()


@pytest.mark.parametrize("learning_rate", (1.0e-6, 3.0e-6, 1.0e-5))
def test_accepts_the_bounded_policy_learning_rate_gate(
    tmp_path: Path, learning_rate: float
) -> None:
    path, text, _ = _write_config(tmp_path)
    text = text.replace(
        "learning_rate = 0.00001", f"learning_rate = {learning_rate:.10f}"
    )
    path.write_text(text, encoding="utf-8")

    config = load_policy_e2e_smoke_run_config(path)

    assert config.optimizer.learning_rate == learning_rate


def test_rejects_an_unplanned_policy_learning_rate(tmp_path: Path) -> None:
    path, text, _ = _write_config(tmp_path)
    path.write_text(
        text.replace("learning_rate = 0.00001", "learning_rate = 0.000002"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="optimizer.learning_rate must be one of"):
        load_policy_e2e_smoke_run_config(path)


def test_selected_sample_dataset_binding_is_exact_and_read_only(tmp_path: Path) -> None:
    path, _, _ = _write_config(tmp_path)
    config = load_policy_e2e_smoke_run_config(path)
    binding = VerlSelectedSampleDatasetBinding.from_run_config(config)

    assert binding.sample_id == SAMPLE_ID
    assert binding.question.endswith("A. first\nB. second")
    assert binding.ground_truth == "B"
    assert binding.repeat_count == 8
    assert binding.as_config()["samples_sha256"] == config.dataset.samples_sha256
    assert "question" not in binding.as_config()
    assert VerlSelectedSampleDatasetBinding.from_config(binding.as_config()) == binding
    assert not config.output.root.exists()


def test_composed_selected_sample_binding_preserves_multiline_text(
    tmp_path: Path,
) -> None:
    path, _, _ = _write_config(tmp_path)
    config = load_policy_e2e_smoke_run_config(path)
    plan = build_policy_e2e_smoke_verl_plan(config)
    upstream_config_dir = (
        Path(__file__).resolve().parents[2]
        / ".deps"
        / "verl"
        / "verl"
        / "trainer"
        / "config"
    )
    composed = compose_upstream_verl_config(plan, config_directory=upstream_config_dir)

    restored = VerlSelectedSampleDatasetBinding.from_config(
        composed.data.tgvf_selected_sample
    )
    assert restored.question == config.dataset.selected_sample.question
    assert restored.ground_truth == config.dataset.selected_sample.ground_truth


def test_verl_actor_batch_mapping_preserves_nontrivial_gradient_accumulation(
    tmp_path: Path,
) -> None:
    path, text, _ = _write_config(tmp_path)
    text = text.replace("global_prompt_batch_size = 4", "global_prompt_batch_size = 8")
    text = text.replace(
        "gradient_accumulation_steps = 1", "gradient_accumulation_steps = 2"
    )
    path.write_text(text, encoding="utf-8")
    config = load_policy_e2e_smoke_run_config(path)
    plan = build_policy_e2e_smoke_verl_plan(config)
    contract = plan.overrides["actor_rollout_ref.rollout.custom"][
        "actor_batch_contract"
    ]

    assert plan.overrides["actor_rollout_ref.actor.ppo_mini_batch_size"] == 8
    assert plan.overrides["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == 1
    assert (
        plan.overrides["actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"] == 8
    )
    assert contract["upstream_internal_mini_batch_size_trajectories"] == 64
    assert contract["derived_gradient_accumulation_steps"] == 2
    assert contract["derived_actor_forward_backward_microbatches"] == 16
    assert contract["optimizer_steps_per_trainer_step"] == 1
