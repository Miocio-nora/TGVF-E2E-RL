"""Fail-closed contract exceptions."""


class TGVFContractError(RuntimeError):
    """Base error for a violated project contract."""


class ContractUnsetError(TGVFContractError):
    """Raised when a required research decision has not been selected."""


class IdentityMismatchError(TGVFContractError):
    """Raised when a component or artifact identity does not match replay."""


class ReplayMismatchError(TGVFContractError):
    """Raised when replay differs from the materialized rollout state."""


class UnsupportedSupportLevelError(TGVFContractError):
    """Raised when a model-family capability is claimed but not implemented."""
