from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from tgvf_rl.evaluation import policy_paired_tgvf_snapshot as implementation


def _fake_run(tmp_path: Path) -> SimpleNamespace:
    config = tmp_path / "run.toml"
    config.write_text("frozen run bytes\n", encoding="utf-8")
    artifact = tmp_path / "adapter.pt"
    artifact.write_bytes(b"adapter-storage")
    return SimpleNamespace(
        run_id="paired-test",
        identity_sha256="1" * 64,
        model=SimpleNamespace(revision_or_path=str(tmp_path / "model")),
        representation=SimpleNamespace(
            artifact_path=artifact,
            artifact_file_sha256=implementation._sha256_file(artifact),
            artifact=SimpleNamespace(sha256="2" * 64),
        ),
    )


def _fake_model(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (model / "model.safetensors.index.json").write_text(
        '{"weight_map":{"model.visual.patch_embed.weight":'
        '"model-00001-of-00001.safetensors",'
        '"model.language_model.layers.0.weight":'
        '"model-00001-of-00001.safetensors"}}\n',
        encoding="utf-8",
    )
    return model


def test_step_zero_materialization_and_lightweight_reload(
    tmp_path: Path, monkeypatch
) -> None:
    run = _fake_run(tmp_path)
    model = _fake_model(tmp_path)
    state = {"query.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)}
    export = SimpleNamespace(state=state, manifest=object())
    monkeypatch.setattr(
        implementation, "load_policy_e2e_smoke_run_config", lambda *args, **kwargs: run
    )
    monkeypatch.setattr(
        implementation,
        "load_rank_zero_adapter_owned_state_export",
        lambda *args, **kwargs: export,
    )
    monkeypatch.setattr(implementation, "state_digest", lambda value: "2" * 64)
    receipt_path = tmp_path / "evaluation/runtime/paired.json"

    receipt = implementation.materialize_paired_tgvf_snapshot(
        policy_config_path=tmp_path / "run.toml",
        optimizer_step=0,
        qwen_model_path=model,
        receipt_path=receipt_path,
    )
    loaded = implementation.load_paired_tgvf_snapshot(receipt_path)

    assert receipt.rp66_kind == "stage1_artifact"
    assert receipt.optimizer_step == 0
    assert loaded.policy_version.weights_sha256 == receipt.combined_weights_sha256
    assert loaded.receipt.qwen_tree_sha256 == receipt.qwen_tree_sha256
    assert torch.equal(loaded.rp66_tensors["query.weight"], state["query.weight"])


def test_qwen_closure_rejects_embedded_rp66_keys(tmp_path: Path) -> None:
    model = _fake_model(tmp_path)
    index = model / "model.safetensors.index.json"
    index.write_text(
        '{"weight_map":{"model.visual.weight":"model-00001-of-00001.safetensors",'
        '"model.language_model.weight":"model-00001-of-00001.safetensors",'
        '"tgvf_adapter.query.weight":"model-00001-of-00001.safetensors"}}\n',
        encoding="utf-8",
    )

    try:
        implementation._validate_qwen_weight_index(model)
    except ValueError as error:
        assert "embeds RP66" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("embedded RP66 key was accepted")
