from __future__ import annotations

import torch

from tgvf_rl.evaluation import policy_paired_qwen_materialization as implementation


def test_qwen_only_partition_requires_exact_base_and_rp66_namespaces() -> None:
    qwen = frozenset(
        {
            "model.visual.patch_embed.weight",
            "model.language_model.layers.0.weight",
        }
    )
    adapter = frozenset({"target_proj.weight", "visual_proj.weight"})
    state = {
        "model.visual.patch_embed.weight": torch.ones(1),
        "model.language_model.layers.0.weight": torch.ones(1),
        "tgvf_adapter.target_proj.weight": torch.ones(1),
        "tgvf_adapter.visual_proj.weight": torch.ones(1),
    }

    selected = implementation._take_qwen_only_state_dict(
        state,
        expected_qwen_keys=qwen,
        expected_adapter_keys=adapter,
    )

    assert frozenset(selected) == qwen
    assert not any(name.startswith("tgvf_adapter.") for name in selected)


def test_qwen_only_partition_rejects_an_unclassified_parameter() -> None:
    try:
        implementation._partition_checkpoint_keys(
            {"model.visual.weight", "foreign.weight", "tgvf_adapter.query.weight"},
            expected_qwen_keys=frozenset({"model.visual.weight"}),
            expected_adapter_keys=frozenset({"query.weight"}),
        )
    except ValueError as error:
        assert "Qwen keys differ" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("an unclassified FSDP parameter was accepted")
