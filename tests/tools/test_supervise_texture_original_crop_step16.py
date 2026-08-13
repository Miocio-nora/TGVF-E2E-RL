from __future__ import annotations

from pathlib import Path
import re
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR = REPOSITORY_ROOT / "tools/supervise_texture_original_crop_step16.sh"


def test_texture_two_arm_supervisor_has_valid_bash_syntax_and_help() -> None:
    subprocess.run(["bash", "-n", str(SUPERVISOR)], check=True)
    completed = subprocess.run(
        [str(SUPERVISOR), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Crop ranks use physical GPUs 0-3" in completed.stdout
    assert "original ranks use physical GPUs 4-7" in completed.stdout
    assert "never kills processes discovered through nvidia-smi" in completed.stdout
    assert "TEXTURE_TWO_ARM_SETUP_MODE" in completed.stdout
    assert "strict (default) or resume" in completed.stdout
    assert "TEXTURE_TWO_ARM_RESUME_VALIDATE_EVIDENCE" in completed.stdout


def test_texture_two_arm_supervisor_pins_cold_jit_runtime_and_durable_closure() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")

    for required in (
        "export CC=/usr/bin/gcc",
        "export CXX=/usr/bin/g++",
        "python312-dev/root/usr/include",
        "nvidia/cublas/include",
        "nvidia/cuda_nvrtc/include",
        "nvidia/cuda_runtime/include",
        ".eval-runtime-python312-dev/lib",
        "mm_encoder_attn_backend",
        "TORCH_SDPA",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "FLASHINFER_WORKSPACE_BASE",
        "VLLM_WORKER_MULTIPROC_METHOD=spawn",
        'socket_tmp=$(mktemp -d "/tmp/t2a-',
        'TMPDIR="$socket_tmp"',
        "ZeroMQ IPC path budget",
    ):
        assert required in source

    assert "unset C_INCLUDE_PATH CPLUS_INCLUDE_PATH CUDA_PATH LD_LIBRARY_PATH" in source
    assert not re.search(r"^export VLLM_ATTENTION_BACKEND=", source, re.MULTILINE)
    assert 'TMPDIR="$cache_root/tmp"' not in source
    assert "original-status" in source
    assert "original-finalize" in source
    assert (
        source.count('"$python_bin" "$repo_root/tools/score_texture_benchmark.py"') == 2
    )
    assert (
        "original_engine_kwargs=${TEXTURE_ORIGINAL_ENGINE_KWARGS_JSON:-"
        '\'{"gpu_memory_utilization":0.8,"max_model_len":32768,'
        '"max_num_batched_tokens":32768,"max_num_seqs":8}\'}'
    ) in source


def test_texture_two_arm_resume_is_explicit_and_binds_existing_setup() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")

    for required in (
        "setup_mode=${TEXTURE_TWO_ARM_SETUP_MODE:-strict}",
        'if [[ "$setup_mode" == strict ]]',
        'if [[ "$setup_mode" == resume ]]',
        "--no-verify-images",
        "stage_resume_validation_evidence",
        "write_resume_setup_evidence",
        "policy-benchmark-tasks.jsonl",
        "frozen-full-model-state/snapshot-manifest.json",
        "frozen-full-model-state/materialization-receipt.json",
        "evaluation-identity.json",
        "resume-static-validation.json",
        "setup/resume-records",
        '"gpu_or_api_used": False',
        '"vllm_engine_constructed": False',
        '"enable_chunked_prefill": False',
        '"gpu_memory_utilization": 0.9',
        '"inference_concurrency_per_gpu": 8',
        '"max_model_len": 32768',
        '"max_num_batched_tokens": 32768',
        '"policy-benchmark-config.json"',
        "argv[:2] != [expected_python, expected_runner]",
    ):
        assert required in source

    strict_block = source[source.index('if [[ "$setup_mode" == strict ]]') :]
    assert "--verify-images" in strict_block
    assert "execute_crop_plan_step materialize" in strict_block
    assert "execute_crop_plan_step prepare" in strict_block
    assert "execute_crop_plan_step validate" in strict_block
    assert "validation_provenance=external_preexisting_evidence" in source
    assert "validation_provenance=control_root_preexisting_evidence" in source
    assert "validation_provenance=executed_by_supervisor" in source
    assert 'skipped_steps.append("crop_static_validate")' in source


def test_texture_two_arm_supervisor_binds_worker_source_closure() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")

    for required in (
        '"$repo_root/src/tgvf_rl"',
        '"$dependency_root/.deps/verl/verl"',
        "compute_source_closure",
        "worker_source_closure",
        "source closure changed during setup",
        "source closure changed before retry",
        "source closure changed during execution",
        "os.link(temporary, path)",
    ):
        assert required in source


def test_texture_two_arm_supervisor_is_fail_closed_about_gpu_ownership() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")

    assert "--query-compute-apps=" in source
    assert 'record["memory_used_mib"] > 16' in source
    assert 'record["utilization_percent"] != 0' in source
    assert "no process was signalled" in source
    assert "terminate_owned_worker" in source
    assert "worker_pid=$!" in source
    assert 'pgid=$(ps -o pgid= -p "$worker_pid"' in source
    assert not re.search(r"\b(?:pkill|killall|fuser)\b", source)
    assert "--gpu-reset" not in source
