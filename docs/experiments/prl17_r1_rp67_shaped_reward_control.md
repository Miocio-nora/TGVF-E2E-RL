# PRL17-R1: frozen-RP67 matched shaped-reward control

## Question

PRL17-R0 showed that the Stage3-derived shaped reward can improve a policy
using a frozen RP66 representation under the larger BS16 × n16, world-size 8
protocol. PRL17-R1 asks one narrower question: does the same RL treatment also
work when the Stage1 representation is RP67?

This is a representation-only follow-up. It is not a new reward experiment,
and it is not a joint-RL update of the Adapter.

## Only intended variable

R0 uses the final RP66 step-2000 Adapter. R1 replaces it with the final RP67
step-2000 Adapter:

- path: `artifacts/representation/RP-67-qwen3-instruct-balanced-t1-image-axis-grounded-2000-gpu01/adapter.pt`;
- artifact SHA-256: `13332865eb30a2b04ce2ee90a9228e490c718e87fa57bc758078cdd28b6f0f68`;
- manifest SHA-256: `2ea098967ba36671d6975a17e3830778d441149c27f5f80e43e78daf818933b1`;
- run identity SHA-256: `0b53d04cf8e4c8b665e76279da1df8d1e6ebabee63318c644a3bff5bad099b44`;
- semantic Adapter-state SHA-256: `f223d1f01b1a188de54b4c6458e1aa456696e566e015fcb570135517848c0256`;
- frozen runtime storage SHA-256: `3f60f36589a3c0f3549c12b949eaabb140f6edfac849aa2b25a623bbcde53a14`.

RP66 and RP67 have the same Adapter contract: 104 BF16 tensors and
72,055,808 parameters with identical names, shapes, and dtypes. RP67 differs
in its Stage1 objective: it adds the correct-image versus donor-image,
image-axis MatrixCE constraint.

The Adapter remains `frozen_adapter`. RL updates the full Qwen3-VL-8B policy,
including its vision path, but does not update RP67. This isolates the effect
of the initial representation.

## Fixed training protocol

Everything below is copied from PRL17-R0:

- model: `Qwen3-VL-8B-Instruct`, native DeepStack enabled;
- data: 77,541 retained T1 samples, the same seed-42 stratified schedule;
- prompt: the same TGVF-only prompt and tool schema, at most six calls;
- batch: 16 prompts × 16 trajectories = 256 trajectories per step;
- topology: world size 8, prompt micro-batch 2 per rank, GA 1;
- optimization: full Qwen update, constant learning rate `1e-6`, eight steps;
- checkpointing: every completed step is recoverable; steps 4, 5, 6, and 8
  are permanent;
- answer judge: the same Qwen2.5-72B text judge and retry/concurrency policy;
- Focus and Grounding rewards: disabled, so no visual judge is loaded.

The executed reward is `R = 2*A_gated + T + P`, where `A_gated` is verified
answer correctness with the established tool-use gate, `T` is the fixed
counterfactual tool-utility reward, and `P` is the protocol-error penalty.

The RP66-derived 128-row utility sidecar is deliberately reused. These labels
are representation-dependent, but regenerating them with RP67 would change
both the representation and the reward. Reuse is therefore necessary for the
strict first comparison. An RP67-rematerialized-label run, if desired, is a
separate experiment.

## Execution and evaluation

Before the formal pilot, a console-only functional canary uses four prompts,
two trajectories per prompt, four GPUs, one optimizer step, and a 512-token
policy budget. It must produce at least one successful TGVF observation and a
complete checkpoint. It is an engineering gate, not a scientific result, and
does not upload to W&B.

The formal run uses all eight GPUs for eight steps. A tmux-owned supervisor
keeps the job independent of the interactive session, resumes only from
complete paired checkpoints after recoverable failures, and launches the
paired evaluation after step 8.

CoreDev-2511 evaluates step 0 and step 8 under the same TGVF protocol. With
eight GPUs the two arms run in parallel. The seven reported datasets are
VStarBench, HRBench4K, BLINK, OCRBench_v2, MMMU-Pro-10c, MathVista-MINI, and
MathVerse-MINI. Step0-to-step8 is the primary RP67 RL effect; R1 versus R0 at
the corresponding step is the representation comparison.

## Interpretation gates

A useful RP67 result requires all of the following:

1. RP67 storage and semantic state remain unchanged throughout RL.
2. Training has finite loss/gradient/reward metrics and no format or tool
   collapse.
3. The step0/step8 evaluator binds the matching Qwen checkpoint and frozen
   RP67 state for each arm.
4. Step8 improves the seven-dataset macro over RP67 step0, or gives a clear
   per-dataset trade-off that is stronger than the RP66 control.

The experiment must not be interpreted from training reward alone. External
CoreDev-2511 accuracy and tool-use health are the final evidence.
