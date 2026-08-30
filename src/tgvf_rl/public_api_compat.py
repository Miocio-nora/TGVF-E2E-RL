"""Safe public-identity rebinding for implementation leaves.

Extracted implementation modules use these helpers to preserve historical
facade and pickle coordinates.  A function is eligible only while its
``__module__`` still names the implementation module.  That ownership gate is
important for dataclass and Protocol classes, whose dictionaries may contain
shared helpers owned by :mod:`dataclasses` or :mod:`typing`.
"""

from __future__ import annotations

from types import FunctionType


def rebind_public_function(
    function: FunctionType,
    *,
    implementation_module: str,
    public_module: str,
    public_name: str | None = None,
    public_qualname: str | None = None,
) -> bool:
    """Rebind one implementation-owned function to its historical identity.

    ``False`` means that the function is not owned by ``implementation_module``
    (including an already-rebound function).  In that case it is left entirely
    unchanged.  Optional names are applied only as part of a successful
    ownership-gated rebind.
    """

    if not isinstance(function, FunctionType):
        raise TypeError("public API compatibility requires a Python function")
    if function.__module__ != implementation_module:
        return False
    function.__module__ = public_module
    if public_name is not None:
        function.__name__ = public_name
    if public_qualname is not None:
        function.__qualname__ = public_qualname
    return True


def rebind_public_class(
    contract_type: type[object],
    *,
    implementation_module: str,
    public_module: str,
) -> bool:
    """Rebind a leaf-owned class and its eligible methods/accessors.

    Class members owned by another module are deliberately skipped.  This
    preserves shared stdlib helpers while retaining the legacy coordinates of
    ordinary methods and property accessors defined by the implementation.
    The operation is idempotent so facade-first and leaf-first import paths
    converge on the same object identity.
    """

    if not isinstance(contract_type, type):
        raise TypeError("public API compatibility requires a class")
    if contract_type.__module__ not in {implementation_module, public_module}:
        return False

    changed = contract_type.__module__ == implementation_module
    contract_type.__module__ = public_module
    for member in vars(contract_type).values():
        functions = (
            (member.fget, member.fset, member.fdel)
            if isinstance(member, property)
            else (member,)
        )
        for function in functions:
            if isinstance(function, FunctionType):
                changed = (
                    rebind_public_function(
                        function,
                        implementation_module=implementation_module,
                        public_module=public_module,
                    )
                    or changed
                )
    return changed


__all__ = ["rebind_public_class", "rebind_public_function"]
