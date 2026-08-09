from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tgvf_rl.framework.verl.policy_weight_sync import (
    PolicyWeightSyncState,
    publish_policy_weight_sync_request,
)
from tgvf_rl.framework.verl.trainable_tgvf_weight_sync import (
    load_latest_trainable_rp66_snapshot,
    split_trainable_rp66_parameter_stream_for_snapshot,
)


@pytest.mark.parametrize("base_sync_done", [False, True])
def test_full_model_stream_sends_qwen_and_publishes_rp66(
    tmp_path: Path, base_sync_done: bool
) -> None:
    environment = {
        "TGVF_POLICY_STATE_DIR": str(tmp_path.resolve()),
        "TGVF_POLICY_RUN_ID": "PRL15-test",
        "TGVF_POLICY_RUN_IDENTITY_SHA256": "1" * 64,
    }
    state = PolicyWeightSyncState.from_environment(environment)
    request = publish_policy_weight_sync_request(state, 3, nonce="test-request")
    qwen = torch.tensor([1.0, 2.0])
    adapter_a = torch.tensor([[3.0]])
    adapter_b = torch.tensor([4.0])

    output = tuple(
        split_trainable_rp66_parameter_stream_for_snapshot(
            (
                ("model.language_model.weight", qwen),
                ("tgvf_adapter.wq.weight", adapter_a),
                ("module.tgvf_adapter.norm.bias", adapter_b),
            ),
            base_sync_done=base_sync_done,
            rank=0,
            world_size=8,
            global_steps=3,
            environment=environment,
        )
    )

    assert output == (("model.language_model.weight", qwen),)
    snapshot = load_latest_trainable_rp66_snapshot(
        state,
        expected_optimizer_step=3,
        expected_request_sha256=request.request_sha256,
    )
    assert set(snapshot.tensors) == {"wq.weight", "norm.bias"}
    assert torch.equal(snapshot.tensors["wq.weight"], adapter_a)
    assert torch.equal(snapshot.tensors["norm.bias"], adapter_b)
