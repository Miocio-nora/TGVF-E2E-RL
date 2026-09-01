"""Dependency-light identity for dynamic global-token policy reduction."""

DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODE = "tgvf_dynamic_global_token_mean"
DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODULE = (
    "tgvf_rl.framework.verl.dynamic_token_actor_loss"
)
DYNAMIC_GLOBAL_TOKEN_LOSS_AGG_MODE = "token-mean"
METHOD_MATRIX_BYPASS_LOSS_REGISTRY_NAME = "bypass_mode"
METHOD_MATRIX_BYPASS_LOSS_MODULE = "tgvf_rl.framework.verl.method_bypass_actor_loss"


__all__ = [
    "DYNAMIC_GLOBAL_TOKEN_LOSS_AGG_MODE",
    "DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODE",
    "DYNAMIC_GLOBAL_TOKEN_POLICY_LOSS_MODULE",
    "METHOD_MATRIX_BYPASS_LOSS_MODULE",
    "METHOD_MATRIX_BYPASS_LOSS_REGISTRY_NAME",
]
