from __future__ import annotations

import dataclasses
import pickle
import sys
from types import FunctionType, ModuleType
import typing

import pytest

from tgvf_rl.public_api_compat import (
    freeze_public_class_annotations,
    rebind_public_class,
    rebind_public_function,
)


_SHARED_STDLIB_FUNCTIONS = (
    dataclasses._dataclass_getstate,  # noqa: SLF001
    dataclasses._dataclass_setstate,  # noqa: SLF001
    typing._no_init_or_replace_init,  # noqa: SLF001
)


def _module(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__dict__["__builtins__"] = __builtins__
    return module


@pytest.mark.parametrize("facade_first", [False, True])
def test_rebinding_is_import_order_independent_and_pickle_safe(
    facade_first: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order = "facade_first" if facade_first else "leaf_first"
    implementation_name = f"tgvf_public_api_compat_{order}_implementation"
    public_name = f"tgvf_public_api_compat_{order}_public"
    implementation = _module(implementation_name)
    public = _module(public_name)
    monkeypatch.setitem(sys.modules, implementation_name, implementation)
    if facade_first:
        monkeypatch.setitem(sys.modules, public_name, public)

    exec(
        """
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True, slots=True)
class Contract:
    value: int

    @property
    def doubled(self) -> int:
        return self.value * 2

    @classmethod
    def from_text(cls, value: str) -> "Contract":
        return cls(int(value))

    @staticmethod
    def normalize(value: int) -> int:
        return abs(value)

class Port(Protocol):
    def consume(self, value: int) -> int: ...

def implementation_function(value: int) -> int:
    return value + 1
""",
        implementation.__dict__,
    )
    contract = implementation.Contract
    port = implementation.Port
    function = implementation.implementation_function

    assert rebind_public_class(
        contract,
        implementation_module=implementation_name,
        public_module=public_name,
    )
    assert rebind_public_class(
        port,
        implementation_module=implementation_name,
        public_module=public_name,
    )
    assert rebind_public_function(
        function,
        implementation_module=implementation_name,
        public_module=public_name,
        public_name="legacy_function",
        public_qualname="legacy_function",
    )

    if not facade_first:
        monkeypatch.setitem(sys.modules, public_name, public)
    public.Contract = contract
    public.Port = port
    public.legacy_function = function

    assert public.Contract is implementation.Contract
    assert public.Port is implementation.Port
    assert public.legacy_function is implementation.implementation_function
    assert pickle.loads(pickle.dumps(contract)) is contract
    assert pickle.loads(pickle.dumps(contract(3))) == contract(3)
    assert pickle.loads(pickle.dumps(port)) is port
    assert pickle.loads(pickle.dumps(function)) is function
    assert contract.__module__ == public_name
    assert contract.doubled.fget.__module__ == public_name
    assert vars(contract)["from_text"].__func__.__module__ == public_name
    assert vars(contract)["normalize"].__func__.__module__ == public_name
    assert contract.from_text("3") == contract(3)
    assert contract.normalize(-3) == 3
    assert port.consume.__module__ == public_name
    assert function.__module__ == public_name
    assert function.__name__ == "legacy_function"
    assert function.__qualname__ == "legacy_function"

    assert not rebind_public_class(
        contract,
        implementation_module=implementation_name,
        public_module=public_name,
    )
    assert not rebind_public_function(
        function,
        implementation_module=implementation_name,
        public_module=public_name,
    )


def test_rebinding_never_mutates_shared_dataclasses_or_typing_functions() -> None:
    before = tuple(
        (function.__module__, function.__name__, function.__qualname__)
        for function in _SHARED_STDLIB_FUNCTIONS
    )

    for function in _SHARED_STDLIB_FUNCTIONS:
        assert isinstance(function, FunctionType)
        assert not rebind_public_function(
            function,
            implementation_module="tgvf_rl.synthetic_implementation",
            public_module="tgvf_rl.synthetic_public",
            public_name="mutated_name",
            public_qualname="mutated_qualname",
        )

    after = tuple(
        (function.__module__, function.__name__, function.__qualname__)
        for function in _SHARED_STDLIB_FUNCTIONS
    )
    assert after == before
    assert {function.__module__ for function in _SHARED_STDLIB_FUNCTIONS} == {
        "dataclasses",
        "typing",
    }


def test_annotation_freezing_supports_leaf_first_type_hints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation_name = "tgvf_public_api_compat_annotations_implementation"
    public_name = "tgvf_public_api_compat_annotations_public"
    implementation = _module(implementation_name)
    monkeypatch.setitem(sys.modules, implementation_name, implementation)

    exec(
        """
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class Payload:
    value: int

class Base:
    inherited: Payload

@dataclass(frozen=True)
class Contract(Base):
    PayloadAlias = Payload
    payload: PayloadAlias
    mode: Literal["strict"]
""",
        implementation.__dict__,
    )
    contract = implementation.Contract
    assert all(isinstance(value, str) for value in contract.__annotations__.values())
    assert isinstance(implementation.Base.__annotations__["inherited"], str)

    assert freeze_public_class_annotations(
        contract,
        implementation_globals=implementation.__dict__,
    )
    assert not any(
        isinstance(value, str) for value in contract.__annotations__.values()
    )
    assert isinstance(implementation.Base.__annotations__["inherited"], str)
    assert not freeze_public_class_annotations(
        contract,
        implementation_globals=implementation.__dict__,
    )
    assert rebind_public_class(
        contract,
        implementation_module=implementation_name,
        public_module=public_name,
    )

    assert public_name not in sys.modules
    assert typing.get_type_hints(contract) == {
        "inherited": implementation.Payload,
        "payload": implementation.Payload,
        "mode": typing.Literal["strict"],
    }
