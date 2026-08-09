# PRL15 Hard-Constraint Audit

Date: 2026-08-10

This audit separates scientific invariants from experiment declarations and
implementation limitations.  A configuration value is not made mandatory
merely because an earlier run happened to use it.

## Decision rule

1. Preserve checks required for the intended mathematical objective or for
   replay/checkpoint correctness.
2. Express one experiment's chosen values in its versioned run declaration and
   Crop/TGVF comparison specification, not as generic engine capabilities.
3. Reject unsupported execution features before rollout/model work whenever
   possible.
4. Do not silently ignore an enabled optimization or diagnostic.

## Current PRL15 classification

| Constraint | Classification | Decision |
|---|---|---|
| Qwen3-VL-8B-Instruct, T1 retained dataset, RP66 artifact, prompt and judge identities | Experiment/data identity | Keep in the versioned run declaration and validate before GPU launch. These identify what was trained; they are not generic engine limits. |
| BS16, n16, LR, optimizer, reward weights, eight-step horizon and checkpoint schedule | Experiment/control declaration | Keep for this pilot and compare against the completed Crop control. Future arms should declare new values without changing low-level replay or sync code. |
| Full-Qwen plus trainable RP66; no policy LoRA | Scientific treatment definition | Keep. Enabling LoRA would change which parameters are optimized and would no longer be the requested Crop comparison. |
| DeepEyes fixed-micro reduction, dynamic batching off, PPO epoch 1, entropy coefficient 0, actor KL loss off, fixed clipping, rollout IS off | Mathematical objective | Keep. These checks prevent a different loss from being labeled as the reproduced DeepEyes objective. Validate at plan construction. |
| Zero rollout staleness and exact behavior token/log-prob/ownership alignment | PPO/replay correctness | Keep. Violations make the importance ratio invalid. |
| Identity sampling transforms and temperature 1.0 | Current exact-replay capability plus current sampling declaration | Keep for this run. Supporting top-p/top-k/penalty replay requires verified operation-order parity; it is not safe to approximate. |
| FSDP2 | Current trainable exact-replay implementation capability | Keep and preflight before rollout. The registered root forward and reshard path currently depend on FSDP2. It is not a theoretical requirement of TGVF. |
| `use_fused_kernels=true`, torch backend | Shared Crop/TGVF execution optimization | Keep equal. TGVF now applies veRL's `FusedLinearForPPO` to injected hidden states at policy-owned positions, avoiding full vocabulary logits while preserving the same values and gradients. |
| Entropy, sum-pi-squared and distillation outputs disabled | Current exact-replay output capability | Keep as an explicit capability declaration. They are not used by this pilot and must fail before rollout if later requested. |
| World size in `{4, 8}` | Current PRL15 arm topology | Keep only in the PRL15 run/launcher. Removed from low-level RP66 weight sync, which now accepts every valid distributed world size. |
| vLLM TP=1 and colocated actor/rollout | Current runtime architecture | Keep as an up-front implementation capability, not a mathematical claim. Relax only after tool-state and RP66 publication parity tests. |
| One replay sequence at a time inside an exact-replay microbatch | Performance implementation limitation | Not theoretically necessary. Keep temporarily for correctness; it is a priority batching optimization after the controlled smoke passes. |
| Four Qwen3 visual streams (main plus three DeepStack branches) and original source pixels for live replay | Qwen3/RP66 structural requirement | Keep. Removing one changes the representation path or loses gradients to the live vision tower. |
| Tensor shapes/dtypes, finite values, autograd presence, role ownership, hashes of realized replay sidecars, sync ACKs and monotonic steps | Runtime correctness | Keep. These detect training the wrong state rather than constrain hyperparameters. |
| `max_samples` disabled in the matched dataset | Crop/TGVF schedule identity | Keep for formal comparison. A smoke subset should be selected by the smoke lifecycle, not by silently changing the formal schedule. |
| W&B formal-only and smoke metrics isolation | Experiment bookkeeping | Keep for the current lifecycle, but it is not a model capability. Smoke remains console-only. |

## Changes made by this audit

- Implemented fused selected-token log-probabilities for trainable TGVF exact
  replay instead of disabling `use_fused_kernels`.
- Preserved the existing logits-returning Qwen replay API for backward
  compatibility.
- Changed the exact replay check from an unconditional fused rejection to a
  capability check on the selected replay port.
- Added worker-build preflight for the pinned veRL torch fused primitive, so an
  incompatible backend fails before rollout.
- Removed the `{4, 8}` restriction from low-level RP66 weight synchronization;
  the current PRL15 launcher still declares the two controlled topologies.

## Verification gates

- Fused and eager selected-token log-probabilities must match in value and in
  gradients to both hidden states and LM-head weights.
- The same parity must hold under CUDA BF16 activations with an FP32 master
  LM-head.
- The immutable PRL14 completion must still reconstruct and pass the original
  Crop compose/preflight path.
- The TGVF one-step GPU smoke must complete an optimizer update, RP66/Qwen
  weight publication and a resumable paired checkpoint before formal launch.

