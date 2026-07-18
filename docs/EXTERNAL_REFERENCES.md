# Controlled External References

Status: **reference identities recorded; no dependency installation authorized**
Recorded: **2026-07-19 JST**

This registry fixes the external sources that may inform the design. A recorded
commit is a review identity, not a production dependency pin and not permission
to copy an external repository wholesale. Any code adapted later requires an
accepted task, a narrow source/symbol record, license review, and parity tests.

## veRL compatibility candidates

```text
repository: https://github.com/verl-project/verl
primary exact main snapshot: e003163181731412595257a72ec173071efb125f
stable comparison: v0.8.0@7aed6b230776f963fa09509c10d9c3a767d1102c
main snapshot observed: 2026-07-19 JST
role: selected framework family; exact production pin pending compatibility spike
dependency status: review candidates only; do not install yet
```

Official point-in-time sources reviewed for the proposed spike include:

- [v0.8.0 release notes](https://github.com/verl-project/verl/releases/tag/v0.8.0);
- [v0.8.0 AgentLoop API](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/verl/experimental/agent_loop/agent_loop.py);
- [v0.8.0 tool loop](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/verl/experimental/agent_loop/tool_agent_loop.py);
- [v0.8.0 tool response schema](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/verl/tools/schemas.py);
- [v0.8.0 rollout-correction contract](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/docs/algo/rollout_corr.md);
- [v0.8.0 Qwen3-VL-8B FSDP2 example](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/examples/grpo_trainer/run_qwen3_vl_8b_fsdp.sh);
- [v0.8.0 multimodal distillation example](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/examples/on_policy_distillation_trainer/run_qwen3_vl_8b_fsdp.sh).

Additional exact-main risk sources are:

- [exact main candidate commit](https://github.com/verl-project/verl/commit/e003163181731412595257a72ec173071efb125f);
- [stable-to-main comparison](https://github.com/verl-project/verl/compare/7aed6b230776f963fa09509c10d9c3a767d1102c...e003163181731412595257a72ec173071efb125f);
- [main full-determinism support matrix](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/docs/advance/determinism.md#L102-L127);
- [main Qwen3-VL visual/DeepStack reconstruction](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/verl/models/transformers/qwen3_vl.py#L205-L324);
- [main vLLM async-server generation boundary](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L531-L641);
- [main SGLang async-server generation boundary](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/verl/workers/rollout/sglang_rollout/async_sglang_server.py#L519-L605);
- [main generic tensor collection helper](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/verl/utils/model.py#L752-L802);
- [main FSDP model-input update seam](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/verl/workers/engine/fsdp/transformer_impl.py#L1023-L1167);
- [main extension guide](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/docs/extend_guide.rst#L47-L263);
- [main Qwen-VL monkey-patch boundary](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/verl/models/transformers/monkey_patch.py#L361-L453);
- [main Qwen2.5-VL-7B FSDP2 example](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh#L1-L151);
- [main FSDP checkpoint manager](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/verl/utils/checkpoint/fsdp_checkpoint_manager.py#L57-L327);
- [main vLLM Docker recipe](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/docker/Dockerfile.stable.vllm#L1-L18);
- [main SGLang Docker recipe](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/docker/Dockerfile.stable.sglang#L1-L4).

The stable candidate exposes useful starting surfaces: FSDP2, official Qwen3-VL
and Qwen2.5-VL examples, token-in/token-out AgentLoop execution, rollout
log-probability fields, dynamic extra fields, and multimodal teacher support.
These are candidate capabilities, not compatibility evidence.

Static review also shows why the spike is required. The default tool response
is limited to text/image/video; the stock multimodal postprocess reconstructs
processor inputs from decoded tokens and ordinary media; and the exposed
rollout server call is oriented around image/video/audio rather than immutable
main-`D`/D-DeepStack bundles. The project must prove a maintained/public latent
adapter path that preserves exact rollout observations and log-probability
identity. Default PIL/processor replay is explicitly unacceptable.

The main snapshot's stock Qwen3-VL forward also regenerates image and DeepStack
embeddings from `pixel_values`. Its generic tensor collector and FSDP
`model_inputs` update provide potentially useful seams, but stock replay is
still recomputation, not exact latent replay. Likewise, rollout-level full
determinism was added only after stable v0.8.0 and the main documentation limits it to
vLLM single-turn: SGLang and multi-turn `tool_agent_loop` are explicitly not
supported. These are hard spike questions, not configuration assumptions.

The main vLLM Dockerfile's comment says `0.20.2` while its version argument says
`0.23.0`; both main backend Dockerfiles still set `VERL_VERSION=v0.7.1`. The
C-STABLE recipes also do not by themselves prove the desired source checkout.
Therefore an image recipe alone does not prove candidate identity. Any approved
environment must verify the exact loaded veRL commit and record all resolved
package/image identities.

The complete proposed task and main-versus-stable decision rule are in
`docs/VERL_COMPATIBILITY_SPIKE_PLAN.md`. Neither candidate is a production pin
until the evidence report is explicitly accepted.

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
