# Working Rules for TGVF End-to-End RL

This repository is the only active implementation workspace for the new
end-to-end RL version.

## Scope isolation

- Keep searches, edits, tests, and commands inside this repository by default.
- Do not run parent-directory searches such as `rg ..` or `find ..`.
- Do not import, symlink, vendor wholesale, or add as a submodule either
  `revisit_vlm` or `revisit_vlm_clean`.
- Legacy inspection must be explicit, read-only, pinned to a file plus commit or
  working-tree hash, and recorded in `docs/LEGACY_REFERENCE.md` before porting.
- External implementation references must be pinned in
  `docs/EXTERNAL_REFERENCES.md`; recording a commit does not authorize vendoring,
  dependency installation, or treating a fork as the production framework.
- Never treat a legacy script name, stage name, checkpoint name, or output path
  as an experiment identity.

## Fixed architectural decisions

- The TGVF model structure is preserved.
- The representation phase has two obligations: `D` contains target-specific
  information, and the frozen/base language model can read that information.
- The legacy representation dataset and selected TGVF/DeepStack model/training
  code may be reused only under the provenance and parity rules. The native
  pipeline is new, and the historical TGVF Adapter checkpoint is reference
  only rather than a direct initialization.
- There is no Stage2 SFT in this version.
- The RL policy initializes from the original Qwen reasoning model, not a
  Golden/Stage2 adapter.
- The primary policy/reference family is Qwen3-VL-8B-Thinking. Qwen2.5-VL is a
  required secondary compatibility family, with its exact size and snapshot
  still explicit configuration. Model-facing code must use a family adapter
  rather than hardcode Qwen3-only tensor or processor behavior.
- Do not claim Qwen2.5-VL end-to-end support from an empty adapter slot. It
  requires a separately identified representation artifact plus native
  transcript, both-provider, main-`D`/model-supported branch, exact-replay, and
  objective fixtures. Missing an equivalent DeepStack path is a compatibility
  blocker; dummy branches and Qwen3 checkpoint reuse are forbidden.
- Both target-conditioning providers are required capabilities:
  contextual hidden state and target token embedding. An experiment selects a
  provider, but the implementation and tests must support both.
- The tokenizer is never resized. Project-specific Protocol-C tokens and their
  embedding/head rows are forbidden.
- The protocol uses Qwen's native tool schema and existing `<tool_call>`,
  `</tool_call>`, `<tool_response>`, `</tool_response>`, `<think>`, `</think>`,
  and native vision tokens.
- The native function name is `tgvf_focus_tool`. The runtime supports repeated
  calls, with a configurable safety cap greater than one.
- The legacy custom Stage3 trainer is not ported. Distributed RL infrastructure
  comes from upstream veRL. FSDP2 support is required; the exact veRL commit,
  rollout backend, sharding, device mesh, worker placement, and parallel
  topology are selected from compatibility and throughput evidence.
- SDPO compatibility is an initial architecture requirement, not a later
  retrofit. The pinned reference is `lasgroup/SDPO` commit
  `7c457fc1b1f636ae794eb0362ba37d4743b06fbc`. Upstream veRL remains the base;
  the SDPO repository's veRL tree is reference-only. The skeleton must reserve
  feedback-conditioned teacher context, token alignment, exact-observation
  teacher replay, distillation targets, objective composition, teacher state,
  and checkpoint/resume interfaces. No SDPO optimizer step is allowed until its
  exact mathematics and parity gate are accepted.
- An optional 72B Qwen answer judge is a separate versioned service role. It is
  never the frozen RL reference policy or the SDPO self-teacher.

## Terminology

Use these names in new code and documents:

- **representation phase**: target-specific/readable `D` learning;
- **TGVF Adapter**: the preserved TGVF model module that consumes target
  conditioning and original-image visual features and produces main `D` plus
  the D-DeepStack branches;
- **policy RL phase**: end-to-end routing, target generation, tool use,
  post-`D` reasoning, and answering;
- **native tool trajectory**: Qwen chat-template serialization using the native
  tool schema.

Do not introduce new components named Stage2 or Stage3. Historical names may
appear only in provenance or comparison text.

Use the full phase names in prose until compact replacements are accepted.
Short package, configuration, and CLI identifiers may be defined separately,
but must not use `stage1`, `stage2`, or `stage3` as current component identities.

## Change discipline

- Start every non-trivial task with a macro plan: objective, files to change,
  files not to touch, unresolved decisions, and verification.
- Do not create implementation code until the corresponding task and interface
  are accepted in `docs/PROJECT_TASK.md`.
- Do not install or pin the production veRL dependency matrix until an approved
  veRL compatibility spike has passed. Candidate versions may be used only in
  the isolated spike environment authorized for that task.
- Do not launch GPU work without a `PLANNED` entry in
  `docs/EXPERIMENT_LEDGER.md` and a complete experiment identity.
- Keep algorithm mathematics explicit: behavior-policy log probabilities,
  reference policy, KL estimator, group standard deviation, clipping,
  sequence/token normalization, and gradient accumulation must be recorded.
- A rollout/replay implementation must preserve the actual sampled behavior
  log probabilities. Setting `old_logprobs = new_logprobs.detach()` is forbidden.
- Preserve log probabilities for every policy-sampled token in every assistant
  turn. Record the rollout policy version, sampling backend/version, sampling
  parameters, logit processors, seed/RNG state, and whether log probabilities
  are measured before or after sampling transforms.
- Policy and reference replay must consume the exact rollout-recorded main `D`,
  D-DeepStack tensors, visual layout, positions, masks, and cache contract.
  Recomputing an observation from an updated policy or the reference model is
  forbidden.
- Rollout and replay must use a deterministic forward state. Policy-adapter
  dropout is zero unless exact RNG/mask replay is explicitly proven. No policy
  update may intervene between sampling and the replay that consumes its
  behavior log probabilities; asynchronous staleness must be bounded and
  recorded.
- The exact GRPO equations are a project artifact, not a library default.
  Population/sample standard deviation, advantage scaling, clipping, KL, and
  token/sequence normalization must be frozen before a trainer is accepted.

## Verification priorities

1. No tokenizer growth and exact native transcript round trip.
2. Numerical parity of the extracted TGVF core with its pinned reference.
3. Target-span extraction and `Hq` identity.
4. Target specificity and causal readability of main `D` and every D-DeepStack
   branch.
5. Template-owned versus policy-sampled token masks, with no duplicate
   `<think>` opener in either assistant turn.
6. Tool-environment rollout and policy/reference-replay logit/logprob parity on
   the exact same recorded observation.
7. Both target-conditioning providers and both required Qwen-VL family adapters
   satisfy their declared fixtures without sharing incompatible checkpoints.
8. GRPO and SDPO loss/gradient/checkpoint parity with their separately approved
   mathematics; pure SDPO and any hybrid are distinct objective identities.
9. High-budget reasoning retention against the original Qwen policy.
