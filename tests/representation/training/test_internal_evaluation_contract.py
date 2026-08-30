from __future__ import annotations

import dataclasses
import pickle
import typing
from types import FunctionType
from typing import get_type_hints

import torch

from tgvf_rl.conditioning.base import TargetConditioningProviderKind
from tgvf_rl.representation import training
from tgvf_rl.representation.training import internal_evaluation as facade
from tgvf_rl.representation.training import internal_evaluation_contract as contract
from tgvf_rl.representation.training.readout import (
    RepresentationVisualTensorBundle,
)


_PUBLIC_MODULE = "tgvf_rl.representation.training.internal_evaluation"
_SHARED_DATACLASS_FUNCTIONS = (
    dataclasses._dataclass_getstate,  # noqa: SLF001
    dataclasses._dataclass_setstate,  # noqa: SLF001
)
_SHARED_TYPING_FUNCTIONS = (
    typing._no_init_or_replace_init,  # noqa: SLF001
)


def _contract_types() -> tuple[type[object], ...]:
    return tuple(
        value
        for name in contract.__all__
        if isinstance((value := getattr(contract, name)), type)
    )


def test_contract_facade_and_package_reexport_exact_objects() -> None:
    for name in contract.__all__:
        assert getattr(facade, name) is getattr(contract, name)
        if name in facade.__all__:
            assert getattr(training, name) is getattr(contract, name)


def test_contract_types_keep_public_pickle_and_type_hint_identity() -> None:
    contract_types = _contract_types()

    assert len(contract_types) == 33
    for contract_type in contract_types:
        assert contract_type.__module__ == _PUBLIC_MODULE
        assert pickle.loads(pickle.dumps(contract_type)) is contract_type
        assert get_type_hints(contract_type) == get_type_hints(
            getattr(facade, contract_type.__name__)
        )
        for member in vars(contract_type).values():
            functions = (
                (member.fget, member.fset, member.fdel)
                if isinstance(member, property)
                else (member,)
            )
            for function in functions:
                if not isinstance(function, FunctionType):
                    continue
                if function in _SHARED_DATACLASS_FUNCTIONS:
                    assert function.__module__ == dataclasses.__name__
                    continue
                if function in _SHARED_TYPING_FUNCTIONS:
                    continue
                assert function.__module__ == _PUBLIC_MODULE
                if function.__annotations__:
                    assert get_type_hints(function) is not None


def test_contract_instances_pickle_through_the_historical_facade() -> None:
    identity = contract.RepresentationInternalEvaluationIdentity(
        evaluation_id="contract-pickle",
        model_identity="model",
        checkpoint_identity="checkpoint",
        data_manifest_sha256="0" * 64,
        prompt_identity="prompt",
        target_conditioning_provider=(
            TargetConditioningProviderKind.TARGET_TOKEN_EMBEDDING
        ),
        random_seed=0,
    )
    restored_identity = pickle.loads(pickle.dumps(identity))

    assert restored_identity == identity
    assert type(restored_identity) is facade.RepresentationInternalEvaluationIdentity

    context = contract.NativeDOnlyContext(
        context_id="context",
        transcript_identity="transcript",
        family="qwen3_vl",
        input_ids=torch.tensor([[1]], dtype=torch.long),
        attention_mask=torch.ones((1, 1), dtype=torch.long),
        position_ids=torch.zeros((3, 1, 1), dtype=torch.long),
        d_positions=(0,),
        image_grid_thw=((1, 1, 1),),
    )
    request = contract.NativeFreeContinuationRequest(
        case_id="case",
        variant="value_a",
        expected_value="A",
        context=context,
        observation_identity="observation",
        observation=RepresentationVisualTensorBundle(
            main=torch.zeros((1, 1, 2)),
            deepstack=(torch.zeros((1, 1, 2)),),
            branch_layers=(1,),
            d_deepstack_active=True,
        ),
    )
    restored_request = pickle.loads(pickle.dumps(request))

    assert type(restored_request) is facade.NativeFreeContinuationRequest
    assert type(restored_request.context) is facade.NativeDOnlyContext
    assert restored_request.case_id == request.case_id
    assert torch.equal(restored_request.context.input_ids, context.input_ids)
    assert torch.equal(restored_request.observation.main, request.observation.main)
