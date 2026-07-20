from __future__ import annotations

from pathlib import Path

import pytest

from tgvf_rl.representation.training.evaluation_runner import (
    load_representation_internal_evaluation_run_config,
)


_SHA = "a" * 64
_COMMIT = "b" * 40


def _config_text(root: Path, *, dirty: str = "false", extra: str = "") -> str:
    return f'''schema_version = "representation-internal-evaluation-run-v1"
run_id = "RP-TEST-INTERNAL-EVAL"
{extra}
[code]
repository = "Miocio-nora/TGVF-E2E-RL"
commit = "{_COMMIT}"
dirty = {dirty}

[source]
training_config_path = "{root / 'training.toml'}"
training_config_sha256 = "{_SHA}"

[artifact]
path = "{root / 'adapter.pt'}"
file_sha256 = "{_SHA}"
manifest_sha256 = "{_SHA}"
expected_run_identity_sha256 = "{_SHA}"
expected_global_step = 2000

[execution]
physical_gpu_id = 3

[evaluation]
evaluation_id = "qwen3-test-v1"
ordered_group_manifest_path = "{root / 'groups.json'}"
ordered_group_manifest_sha256 = "{_SHA}"
counterfactual_manifest_path = "{root / 'counterfactual.json'}"
counterfactual_manifest_sha256 = "{_SHA}"
report_path = "{root / 'report.json'}"
random_seed = 42
max_new_tokens = 64
eos_token_ids = [151645]
'''


def test_loads_complete_evaluation_only_identity(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.toml"
    path.write_text(_config_text(tmp_path), encoding="utf-8")

    config = load_representation_internal_evaluation_run_config(path)

    assert config.run_id == "RP-TEST-INTERNAL-EVAL"
    assert config.code_commit == _COMMIT
    assert config.expected_global_step == 2000
    assert config.physical_gpu_id == 3
    assert config.evaluation.enabled is True
    assert config.evaluation.eos_token_ids == (151645,)


def test_rejects_dirty_formal_evaluation_code(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.toml"
    path.write_text(_config_text(tmp_path, dirty="true"), encoding="utf-8")

    with pytest.raises(ValueError, match="dirty=false"):
        load_representation_internal_evaluation_run_config(path)


def test_rejects_unknown_top_level_identity_field(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.toml"
    path.write_text(_config_text(tmp_path, extra='surprise = "no"'), encoding="utf-8")

    with pytest.raises(ValueError, match="fields differ"):
        load_representation_internal_evaluation_run_config(path)
