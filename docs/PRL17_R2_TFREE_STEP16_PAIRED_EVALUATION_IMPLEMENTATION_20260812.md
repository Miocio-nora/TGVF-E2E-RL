# PRL17-R2 T-free Step 0/8/16 Paired Evaluation Implementation

Date: 2026-08-12
Status: implementation and CPU tests complete; GPU evaluation not started by this change

## 1. Purpose

The existing RP67 T-free Step 0/8 result used `temperature=1`, but its
content-addressed RNG inherited the arm-specific evaluation ID and policy
checkpoint identity. Thus the old Step 0 and Step 8 scores did not use common
random numbers. This implementation adds a fresh three-arm CoreDev-2511 run
for Step 0, Step 8, and Step 16 under one explicit paired-seed contract.

This remains the canonical stochastic policy evaluation:

- `temperature=1.0`;
- `top_p=1.0`, `top_k=-1`, `min_p=0.0`;
- `do_sample=true`;
- all response-budget, stop/EOS, prompt, tool-schema, and tool-call limits are
  inherited unchanged from the frozen RP67 T-free training config.

No temp-zero implementation is included. No training sampler or training RNG
code is modified.

## 2. Fixed identity

| Field | Value |
|---|---|
| evaluation ID | `PRL17-R2-FROZEN-RP67-TFREE-COREDEV2511-STEP0-STEP8-STEP16-PAIRED-SEED-V1` |
| output root | `artifacts/policy/PRL-17-R2-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-novisual-8step-ws8/evaluation/PRL17-R2-FROZEN-RP67-TFREE-COREDEV2511-STEP0-STEP8-STEP16-PAIRED-SEED-V1` |
| policy config SHA-256 | `3820cf64ddf5e7cc825f6596f5a5ca02f4234fa65d0e67e3f8ec0e906e78593d` |
| task manifest SHA-256 | `3f69119d24867c3f3210c8b01eb71304247725ddaf9ca983d2b41c2885403cbc` |
| protocol SHA-256 | `e82f05a663928df20e5a757c2de14264c990cc04cb9bf4985e23f1e90e257a25` |
| master seed | `42` |
| seed namespace | `coredev2511-official-v1/rp67-tfree/step0-step8-step16/temp1/seed42/v1` |
| arms | Step 0, Step 8, Step 16 |

The plan is
`configs/evaluation/prl17_r2_frozen_rp67_tfree_step0_step8_step16_paired_seed_coredev2511_plan.json`.

## 3. Paired RNG definition

For sample `i`, rollout `r`, and assistant turn `t`, the RNG state is derived
from the canonical serialization of:

```text
(master_seed,
 seed_namespace,
 task_manifest_sha256,
 protocol_sha256,
 sample_id=i,
 rollout_index=r,
 assistant_turn_index=t)
```

It deliberately excludes:

- evaluation ID and arm name;
- optimizer step;
- checkpoint and policy-weight hashes;
- evolving prompt-token hash.

Therefore the same sample and assistant-turn index use the same vLLM seed at
Step 0, Step 8, and Step 16, even when the policy bytes or preceding generated
text differ. Different samples, turns, rollouts, namespaces, manifests, or
protocols receive different streams.

The ordinary `ContentAddressedVLLMTurnRNG` used by training and legacy
evaluation is unchanged. The paired RNG is selected only when an evaluation
config contains the explicit `paired_seed_namespace` field.

## 4. Resume and cache isolation

The paired run cannot silently reuse the previous T-free Step 0/8 inference:

1. it has a new evaluation ID and output root;
2. the immutable evaluation identity contains the complete paired RNG
   contract;
3. every result row contains that RNG contract plus its per-sample stream
   identity SHA-256;
4. result validation rejects rows produced under the legacy RNG identity;
5. materialized arm configs must all contain the exact same paired namespace.

Within the new output root, interrupted inference and scoring remain resumable
under the existing rank-JSONL and official-scoring rules.

## 5. Automatic execution

The single entry point is:

```bash
tools/supervise_prl17_r2_tfree_step0_step8_step16_paired_evaluation.sh
```

It performs the following sequence without further user input:

1. wait for a complete Step 16 checkpoint and RP67 runtime snapshot;
2. wait until GPUs 0--7 are released by continuation training;
3. materialize exact Step 0, Step 8, and Step 16 Qwen/RP67 pairs;
4. run Step 0 and Step 8 concurrently on four GPUs each;
5. run Step 16 on the released GPUs;
6. materialize the official CoreDev-2511 scoring views;
7. start the pinned local Qwen2.5-72B judge and score all seven suites;
8. write `paired-summary.json` and `evaluation-complete` only after all arms
   are complete.

The continuation supervisor may directly `exec` this script after its Step 16
closure. Estimated three-arm evaluation time is approximately 75--90 minutes,
subject to output length and semantic-judge load.

For a non-GPU identity/plan check:

```bash
.venv312/bin/python tools/run_prl15_paired_evaluation.py \
  --plan configs/evaluation/prl17_r2_frozen_rp67_tfree_step0_step8_step16_paired_seed_coredev2511_plan.json \
  --mode prepare \
  --gpu-ids 0 1 2 3 4 5 6 7
```

Do not run `prepare` before Step 16 exists; the formal supervisor is the
recommended entry.

## 6. Tests

The behavior tests prove:

- the same sample/turn receives the same seed across evaluation IDs, policy
  steps, checkpoint weights, and prompt-token sequences;
- sample, turn, and namespace changes partition the stream;
- a legacy-RNG result row cannot resume into the paired evaluation identity;
- the new plan remains canonical temp=1 and contains all three ordered arms;
- the supervisor has a fresh output identity, waits for the final arm, and
  uses all eight GPUs.

Targeted command:

```bash
.venv312/bin/python -m pytest -q \
  tests/evaluation/test_policy_paired_seed.py \
  tests/evaluation/test_prl17_r2_step16_paired_evaluation.py \
  tests/evaluation/test_prl15_paired_evaluation_executor.py \
  tests/evaluation/test_policy_benchmark_config.py
```
