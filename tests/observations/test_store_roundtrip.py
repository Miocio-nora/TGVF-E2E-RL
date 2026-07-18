from __future__ import annotations

import pytest
import torch

from tgvf_rl.contracts.errors import ReplayMismatchError
from tgvf_rl.observations.store import ObservationStore
from tests.support import populated_observation_store


def test_store_is_bit_preserving_and_returns_clones() -> None:
    store, handle = populated_observation_store()
    record = store.resolve_record(handle)
    first = store.resolve_verified(record.payload.main_d)
    assert first.dtype is torch.bfloat16
    first.zero_()
    second = store.resolve_verified(record.payload.main_d)
    assert torch.count_nonzero(second).item() > 0


def test_store_checkpoint_round_trip_keeps_record_and_tensors() -> None:
    store, handle = populated_observation_store()
    restored = ObservationStore.from_checkpoint_state(store.checkpoint_state())
    record = restored.resolve_record(handle)
    torch.testing.assert_close(
        restored.resolve_verified(record.payload.main_d),
        store.resolve_verified(record.payload.main_d),
        rtol=0,
        atol=0,
    )


def test_unknown_observation_fails_closed() -> None:
    store, handle = populated_observation_store()
    bad = type(handle)("missing", handle.record_sha256)
    with pytest.raises(ReplayMismatchError, match="unknown observation"):
        store.resolve_record(bad)
