from __future__ import annotations

import hashlib

import torch

from tgvf_rl.contracts.identity import ArtifactIdentity, ModelIdentity, PolicyVersion
from tgvf_rl.contracts.tensors import TensorPayloadSet
from tgvf_rl.observations.schema import (
    CacheContract,
    ConditionProvenance,
    DeepStackBranchRecord,
    FocusedObservationRecord,
    ObservationMasks,
    SourceVisualState,
    VisualLayout,
)
from tgvf_rl.observations.store import ObservationHandle, ObservationStore


SHA0 = "0" * 64
SHA1 = "1" * 64
SHA2 = "2" * 64


def policy_version() -> PolicyVersion:
    return PolicyVersion("smoke", 0, SHA0)


def populated_observation_store() -> tuple[ObservationStore, ObservationHandle]:
    store = ObservationStore()
    premerge = store.put_tensor(
        "source.premerge", torch.arange(16, dtype=torch.float32).view(4, 4)
    )
    premerge_branch = store.put_tensor("source.deepstack.8", torch.ones(4, 4))
    merged = store.put_tensor(
        "source.merged", torch.arange(16, dtype=torch.float32).view(4, 4)
    )
    merged_branch = store.put_tensor("source.merged.deepstack.8", torch.ones(4, 4))
    main_d = store.put_tensor(
        "observation.main_d", torch.arange(8, dtype=torch.bfloat16).view(2, 4)
    )
    branch_d = store.put_tensor("observation.deepstack.8", torch.full((2, 4), 2.0))
    positions = store.put_tensor(
        "observation.position_ids", torch.arange(12).view(1, 12)
    )
    attention = store.put_tensor(
        "observation.attention_mask", torch.ones(1, 12, dtype=torch.bool)
    )
    visibility = store.put_tensor(
        "observation.visibility", torch.ones(1, 12, dtype=torch.bool)
    )
    key_block = store.put_tensor(
        "observation.original_key_block", torch.zeros(1, 1, 12, 12, dtype=torch.bool)
    )
    merger = ArtifactIdentity("qwen", "deepstack_merger_0", "fixture", SHA1)
    record = FocusedObservationRecord(
        schema_version="focused-observation-v1",
        observation_id="observation-0",
        call_index=0,
        model=ModelIdentity("qwen3_vl", "fixture", "/fixture", 151669, SHA2),
        representation=ArtifactIdentity("tgvf", "adapter", "fixture", SHA0),
        condition=ConditionProvenance(
            provider="contextual_hidden_state",
            sampled_target_text_sha256=hashlib.sha256(b"red label").hexdigest(),
            sampled_target_token_start=1,
            sampled_target_token_end=3,
            source_input_ids_sha256=SHA0,
            trajectory_ids=("smoke/sample/0/group",),
            call_indices=(0,),
            hidden_layer=18,
            policy_version=policy_version(),
        ),
        source_visual=SourceVisualState(
            image_sha256=SHA2,
            premerge_main=premerge,
            premerge_deepstack=(premerge_branch,),
            merged_main=merged,
            merged_deepstack=(merged_branch,),
            image_grid_thw=(1, 4, 4),
            spatial_merge_size=2,
        ),
        payload=TensorPayloadSet(
            main_d=main_d,
            deepstack=(branch_d,),
            position_ids=positions,
            attention_mask=attention,
        ),
        branches=(DeepStackBranchRecord(8, branch_d, (6, 7), merger),),
        layout=VisualLayout(
            sequence_length=12,
            original_image_positions=(1, 2, 3, 4),
            d_positions=(6, 7),
            deepstack_branch_layers=(8,),
            deepstack_injection_positions=((6, 7),),
        ),
        masks=ObservationMasks(visibility, visibility, visibility, key_block),
        cache=CacheContract("no_cache", 0, None, None, True, 0.0),
    )
    return store, store.put(record)
