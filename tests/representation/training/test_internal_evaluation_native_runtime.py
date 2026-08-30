from __future__ import annotations

import pickle
from types import FunctionType
from typing import get_type_hints

import pytest

from tgvf_rl.representation import training
from tgvf_rl.representation.training import internal_evaluation as facade
from tgvf_rl.representation.training import (
    internal_evaluation_native_runtime as native_runtime,
)


_PUBLIC_MODULE = "tgvf_rl.representation.training.internal_evaluation"
_RUNTIME_SYMBOLS = (
    "InjectedNativeCounterfactualEvaluator",
    "_greedy_token_id",
    "create_injected_native_counterfactual_evaluator",
    "_validate_materialized_request",
    "_validated_generated_ids",
    "_evaluate_native_case",
    "_evaluate_target_presence_case",
    "_summarize_counterfactuals",
    "_zero_visual_bundle",
    "_summarize_target_presence",
)


def test_native_runtime_facade_reexports_exact_objects() -> None:
    for name in _RUNTIME_SYMBOLS:
        assert getattr(facade, name) is getattr(native_runtime, name)
    for name in native_runtime.__all__:
        assert getattr(training, name) is getattr(native_runtime, name)


def test_native_runtime_preserves_function_and_pickle_coordinates() -> None:
    evaluator_type = native_runtime.InjectedNativeCounterfactualEvaluator

    assert evaluator_type.__module__ == _PUBLIC_MODULE
    assert pickle.loads(pickle.dumps(evaluator_type)) is evaluator_type
    for member in vars(evaluator_type).values():
        functions = (
            (member.fget, member.fset, member.fdel)
            if isinstance(member, property)
            else (member,)
        )
        for function in functions:
            if not isinstance(function, FunctionType):
                continue
            assert function.__module__ == _PUBLIC_MODULE
            if function.__annotations__:
                assert get_type_hints(function) is not None

    for name in _RUNTIME_SYMBOLS[1:]:
        function = getattr(native_runtime, name)
        assert function.__module__ == _PUBLIC_MODULE
        assert pickle.loads(pickle.dumps(function)) is function
        assert get_type_hints(function) is not None


def test_runner_keeps_native_helper_globals_patchable_through_the_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_globals = facade.run_representation_internal_evaluation.__globals__

    for name in (
        "_evaluate_native_case",
        "_evaluate_target_presence_case",
        "_summarize_counterfactuals",
        "_summarize_target_presence",
    ):
        original = getattr(native_runtime, name)
        replacement = object()
        assert runner_globals[name] is original
        monkeypatch.setattr(facade, name, replacement)
        assert runner_globals[name] is replacement
        monkeypatch.setattr(facade, name, original)
