# Controlled External References

Status: **reference identities recorded; no dependency installation authorized**
Recorded: **2026-07-19 JST**

This registry fixes the external sources that may inform the design. A recorded
commit is a review identity, not a production dependency pin and not permission
to copy an external repository wholesale. Any code adapted later requires an
accepted task, a narrow source/symbol record, license review, and parity tests.

## SDPO

```text
repository: https://github.com/lasgroup/SDPO
review commit: 7c457fc1b1f636ae794eb0362ba37d4743b06fbc
paper: https://arxiv.org/abs/2601.20802v2
observed: 2026-07-19 JST
role: algorithm and reference-implementation source
dependency status: reference only; do not install or vendor
```

The reference implements Self-Distilled Policy Optimization on a veRL-derived
tree. It conditions the current model on feedback to construct a self-teacher;
this is distinct from an external answer judge. Its exposed design space
includes full-logit or top-k distillation, feedback/reprompt construction,
importance weighting, EMA or trust-region teacher regularization, and teacher
checkpoint state.

The new project must be SDPO-compatible from its first framework skeleton. In
particular, framework-neutral trajectory/objective records must reserve:

- feedback and successful-demonstration identity;
- a separately rendered teacher context and its token alignment;
- teacher/student token masks over multi-turn multimodal trajectories;
- teacher logits/log probabilities or a versioned approximation artifact;
- current-policy self-teacher identity plus EMA or trust-region regularization
  state;
- objective composition, metrics, checkpoint, and resume state.

The commit above fixes the reference implementation. It does **not** yet freeze
our exact SDPO equations, teacher regularization, reprompt template, feedback
policy, full-logit/top-k choice, importance weighting, or GRPO/SDPO composition.
Those values must pass the SDPO mathematics and parity gate before an optimizer
step uses SDPO.

Upstream veRL remains the base infrastructure. The SDPO repository's bundled or
modified veRL tree is not adopted wholesale. The compatibility spike must prove
that the required SDPO seams can be implemented through maintained/public
extension points without changing the exact TGVF trajectory and observation
contracts.

Known reference-implementation gaps that the spike must test explicitly:

- its distillation actor asserts that multimodal inputs are unsupported;
- its teacher reprompt path assumes a simplified text prompt/response rather
  than a complete repeated-tool trajectory;
- it depends on veRL's legacy worker path and does not establish the new-engine
  integration required by a current upstream pin;
- its shipped path does not prove FSDP2 SDPO, LoRA-to-teacher parameter mapping,
  coexistence with a separate KL reference, or strict EMA-teacher save/resume.

These gaps justify an early neutral interface; they are not permission to
silently reduce TGVF teacher replay to text-only distillation.

## DeepEyes

```text
repository: https://github.com/Visual-Agent/DeepEyes
review commit: 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1
observed: 2026-07-19 JST
role: agentic multimodal RL design reference
dependency status: reference only; do not install or vendor
```

Permitted reference topics are:

- veRL-based multi-turn agent-loop and tool-environment boundaries;
- dynamic multimodal observations between assistant turns;
- per-sample tool routing and mixed agent/non-agent data;
- interleaved Qwen-VL M-RoPE, policy-loss masks, and multimodal batching;
- reward/verifier routing, data normalization, tracing, and evaluation patterns;
- serving an independent large Qwen model as an optional answer judge.

DeepEyes uses Qwen2.5-VL policy models and documents
`Qwen/Qwen2.5-72B-Instruct` as an LLM-as-judge example. The user has now fixed
our secondary policy model to `Qwen/Qwen2.5-VL-7B-Instruct` and approved
reserving—without first-pilot deployment—the 72B judge provider. Judge prompt,
reward role, and deployment topology remain unset.

Forbidden inheritance includes its crop/zoom tool identity, rendered prompts,
dataset assumptions, reward coefficients, asynchronous-staleness behavior, and
any logprob/replay convention that has not passed this project's stricter
behavior-logprob and exact-`D` gates. `tgvf_focus_tool` and native TGVF
observations remain authoritative.

## Model roles and current identities

### Primary policy/reference family

```text
family: Qwen3-VL
initial size/variant: 8B Thinking
accepted stable local path:
  /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking
path presence checked: 2026-07-19 JST
full weight-directory hash: intentionally not required
```

The local path came from the pinned legacy README inspection recorded in
`docs/LEGACY_REFERENCE.md`. The user confirmed that this directory is stable and
that full model-directory or weight-shard hashing is unnecessary. Experiments
record the exact model name and absolute path. Processor, tokenizer, chat
template, and native token serialization still require exact golden fixtures
and fixture hashes because they define the protocol.

### Secondary policy compatibility family

```text
model ID: Qwen/Qwen2.5-VL-7B-Instruct
official model card: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
local/runtime path: [TBD]
role: required model-adapter compatibility target, not the first policy run
```

The 7B variant is fixed because it is the closest Qwen2.5-VL scale to the
primary 8B policy and is the main policy configuration documented by the pinned
DeepEyes reference. This is a model choice, not a claim that its TGVF/DeepStack
path already works.

New model-facing interfaces must not hardcode Qwen3-only class names, tensor
field names, processor behavior, DeepStack branch layout, or M-RoPE assembly.
Family-specific behavior belongs behind a versioned Qwen VLM adapter. Supporting
both families means that both adapters and their fixtures are required before
the compatibility claim is made; it does not mean one representation
checkpoint can be shared between model families.

There are two explicit support levels:

1. the initial skeleton must be family-neutral and prove a
   `Qwen/Qwen2.5-VL-7B-Instruct` processor/transcript/forward adapter fixture;
2. an end-to-end support claim for that model additionally requires a separately
   trained family-specific representation artifact, both condition providers,
   native multi-call main-`D` and model-supported branch injection, exact replay,
   and objective fixtures. If the selected model lacks an equivalent DeepStack
   path, that is a compatibility blocker until an accepted mapping exists; dummy
   branches and Qwen3 artifact reuse are forbidden.

### Optional answer judge

```text
reserved model: Qwen/Qwen2.5-72B-Instruct
exact local path/service/snapshot: [TBD]
role: optional semantic answer verifier only; disabled for first pilot
```

The judge is independently versioned and calibrated. It is not the frozen RL
reference policy, the SDPO self-teacher, or a replacement for executable
verifiers.
