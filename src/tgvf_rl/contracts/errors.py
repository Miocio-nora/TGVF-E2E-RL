"""Fail-closed contract exceptions."""

from __future__ import annotations

from collections.abc import Mapping


class TGVFContractError(RuntimeError):
    """Base error for a violated project contract."""


class ContractUnsetError(TGVFContractError):
    """Raised when a required research decision has not been selected."""


class IdentityMismatchError(TGVFContractError):
    """Raised when a component or artifact identity does not match replay."""


class ReplayMismatchError(TGVFContractError):
    """Raised when replay differs from the materialized rollout state."""


class PolicyOutputContractError(ReplayMismatchError):
    """A sample-local model output that cannot form a legal policy turn.

    This is deliberately narrower than :class:`ReplayMismatchError`.  Callers
    may turn this exception into an explicit failed sample, while identity,
    transport, replay, and artifact mismatches must continue to fail closed.
    The diagnostic mapping must contain only bounded, non-secret audit data;
    sampled text belongs in neither the exception nor durable failure rows.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        diagnostic: Mapping[str, object],
    ) -> None:
        if not isinstance(message, str) or not message:
            raise ValueError("policy-output contract message must be non-empty")
        if (
            not isinstance(code, str)
            or not code
            or code.strip() != code
            or any(character.isspace() for character in code)
        ):
            raise ValueError("policy-output contract code must be canonical")
        if not isinstance(diagnostic, Mapping):
            raise TypeError("policy-output contract diagnostic must be a mapping")
        super().__init__(message)
        self.code = code
        self.diagnostic = dict(diagnostic)


class UnsupportedSupportLevelError(TGVFContractError):
    """Raised when a model-family capability is claimed but not implemented."""


class RecoverableToolExecutionError(RuntimeError):
    """An expected tool/environment failure that may become a tool error turn."""
