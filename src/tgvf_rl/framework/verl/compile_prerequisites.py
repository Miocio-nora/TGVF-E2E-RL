"""Compatibility facade for Policy compile-prerequisite contracts.

The implementation is intentionally owned by the dependency-light
``tgvf_rl.ops.policy_compile_prerequisites`` leaf.  Re-export the exact objects
here so historical imports and pickle coordinates remain valid without a
second implementation.
"""

from tgvf_rl.ops.policy_compile_prerequisites import (
    POLICY_COMPILE_PREREQUISITE_BINDING_SCHEMA,
    POLICY_COMPILE_PREREQUISITE_CLOSURE_POLICY,
    POLICY_COMPILE_PREREQUISITE_MANIFEST_SCHEMA,
    POLICY_COMPILE_PREREQUISITE_MISSING_BLOCKER,
    POLICY_COMPILE_PREREQUISITE_RECEIPT_SCHEMA,
    POLICY_COMPILE_PREREQUISITE_RESIDUAL_BLOCKER,
    POLICY_COMPILE_PREREQUISITE_SYSTEM_RESIDUAL,
    PolicyCompilePrerequisiteBinding,
    PolicyCompilePrerequisiteFile,
    PolicyCompilePrerequisiteFileReceipt,
    PolicyCompilePrerequisiteReceipt,
    load_policy_compile_prerequisite_manifest,
    materialize_policy_compile_prerequisite_receipt,
    preflight_policy_compile_prerequisites,
    verify_policy_compile_prerequisite_receipt,
    verify_policy_compile_prerequisites_from_environment,
)

__all__ = [
    "POLICY_COMPILE_PREREQUISITE_BINDING_SCHEMA",
    "POLICY_COMPILE_PREREQUISITE_CLOSURE_POLICY",
    "POLICY_COMPILE_PREREQUISITE_MANIFEST_SCHEMA",
    "POLICY_COMPILE_PREREQUISITE_MISSING_BLOCKER",
    "POLICY_COMPILE_PREREQUISITE_RECEIPT_SCHEMA",
    "POLICY_COMPILE_PREREQUISITE_RESIDUAL_BLOCKER",
    "POLICY_COMPILE_PREREQUISITE_SYSTEM_RESIDUAL",
    "PolicyCompilePrerequisiteBinding",
    "PolicyCompilePrerequisiteFile",
    "PolicyCompilePrerequisiteFileReceipt",
    "PolicyCompilePrerequisiteReceipt",
    "load_policy_compile_prerequisite_manifest",
    "materialize_policy_compile_prerequisite_receipt",
    "preflight_policy_compile_prerequisites",
    "verify_policy_compile_prerequisite_receipt",
    "verify_policy_compile_prerequisites_from_environment",
]
