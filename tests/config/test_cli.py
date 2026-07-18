from __future__ import annotations

from pathlib import Path

from tgvf_rl.cli import main, validate_smoke_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_bounded_fsdp2_config_is_explicit_and_valid() -> None:
    config = validate_smoke_config(REPOSITORY_ROOT / "configs/smoke/fsdp2.toml")
    assert config["stack"]["physical_gpu_ids"] == [2, 3]
    assert config["stack"]["vllm_enable_mm_embeds"] is True
    assert config["objective"]["production_rl"] is False
    assert config["checkpoint"]["contents"] == ["model", "optimizer", "extra"]


def test_cli_can_print_static_compatibility_contract(capsys) -> None:
    assert main(["compat-info"]) == 0
    output = capsys.readouterr().out
    assert '"rollout_backend"' not in output
    assert '"verl_candidate_commit"' in output
