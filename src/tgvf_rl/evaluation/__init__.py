"""Identity-safe external evaluation contracts."""

from .vlmevalkit import (
    COREDEV_2511,
    COREDEV_2511_MANIFEST_ID,
    COREDEV_2511_MANIFEST_SHA256,
    COREDEV_2511_SOURCE_FILE_SHA256,
    SHARED_BENCHMARK_ROOT,
    VLMEVALKIT_REVIEW_COMMIT,
    CoreDev2511Spec,
    CoreDev2511Manifest,
    CoreDevManifestEntry,
    CoreDevSliceSpec,
    TGVFPolicyEvaluationResult,
    VLMEvalKitLaunchPlan,
    load_coredev_2511_manifest,
)

__all__ = [
    "COREDEV_2511",
    "COREDEV_2511_MANIFEST_ID",
    "COREDEV_2511_MANIFEST_SHA256",
    "COREDEV_2511_SOURCE_FILE_SHA256",
    "SHARED_BENCHMARK_ROOT",
    "VLMEVALKIT_REVIEW_COMMIT",
    "CoreDev2511Spec",
    "CoreDev2511Manifest",
    "CoreDevManifestEntry",
    "CoreDevSliceSpec",
    "TGVFPolicyEvaluationResult",
    "VLMEvalKitLaunchPlan",
    "load_coredev_2511_manifest",
]
