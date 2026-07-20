# Controlled External References

Status: **bounded compatibility pin accepted; production training lock not selected**
Recorded: **2026-07-19 JST**

This registry fixes the external sources that may inform the design. A recorded
commit is a review identity, not a production dependency pin and not permission
to copy an external repository wholesale. Any code adapted later requires an
accepted task, a narrow source/symbol record, license review, and parity tests.

## veRL pinned compatibility candidate

```text
repository: https://github.com/verl-project/verl
exact spike snapshot: e003163181731412595257a72ec173071efb125f
main snapshot observed: 2026-07-19 JST
runtime: upstream veRL exact snapshot + vLLM + FSDP2 only
role: accepted bounded framework/runtime compatibility pin
dependency status: resolved compatibility lock accepted; production placement,
  topology, objectives, and training scale remain open
```

`v0.8.0@7aed6b230776f963fa09509c10d9c3a767d1102c` is retained only
as source-history provenance. It is not a runtime comparison, fallback, install,
or GPU cell. SGLang is likewise explicitly outside this spike and first
production implementation.

Official point-in-time sources reviewed for the approved spike include:

- [v0.8.0 release notes](https://github.com/verl-project/verl/releases/tag/v0.8.0);
- [v0.8.0 AgentLoop API](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/verl/experimental/agent_loop/agent_loop.py);
- [v0.8.0 tool loop](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/verl/experimental/agent_loop/tool_agent_loop.py);
- [v0.8.0 tool response schema](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/verl/tools/schemas.py);
- [v0.8.0 rollout-correction contract](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/docs/algo/rollout_corr.md);
- [v0.8.0 Qwen3-VL-8B FSDP2 example](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/examples/grpo_trainer/run_qwen3_vl_8b_fsdp.sh);
- [v0.8.0 multimodal distillation example](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/examples/on_policy_distillation_trainer/run_qwen3_vl_8b_fsdp.sh).

Additional exact-main risk sources are:

- [exact main candidate commit](https://github.com/verl-project/verl/commit/e003163181731412595257a72ec173071efb125f);
- [main full-determinism support matrix](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/docs/advance/determinism.md#L102-L127);
- [main Qwen3-VL visual/DeepStack reconstruction](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/verl/models/transformers/qwen3_vl.py#L205-L324);
- [main vLLM async-server generation boundary](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L531-L641);
- [main generic tensor collection helper](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/verl/utils/model.py#L752-L802);
- [main FSDP model-input update seam](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/verl/workers/engine/fsdp/transformer_impl.py#L1023-L1167);
- [main extension guide](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/docs/extend_guide.rst#L47-L263);
- [main Qwen-VL monkey-patch boundary](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/verl/models/transformers/monkey_patch.py#L361-L453);
- [main Qwen2.5-VL-7B FSDP2 example](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh#L1-L151);
- [main FSDP checkpoint manager](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/verl/utils/checkpoint/fsdp_checkpoint_manager.py#L57-L327);
- [main vLLM Docker recipe](https://github.com/verl-project/verl/blob/e003163181731412595257a72ec173071efb125f/docker/Dockerfile.stable.vllm#L1-L18).

The historical v0.8 source exposes useful starting surfaces: FSDP2, official Qwen3-VL
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
determinism was added only after v0.8.0 and the main documentation limits it to
vLLM single-turn; multi-turn `tool_agent_loop` is explicitly not supported.
These are hard spike questions, not configuration assumptions.

The main vLLM Dockerfile's comment says `0.20.2` while its version argument says
`0.23.0`, and it still sets `VERL_VERSION=v0.7.1`.
Therefore an image recipe alone does not prove candidate identity. Any approved
environment must verify the exact loaded veRL commit and record all resolved
package/image identities.

The bounded task is closed in `docs/VERL_COMPATIBILITY_REPORT.md`. C-MAIN is the
accepted framework compatibility revision; that result does not turn the
environment into a production-training topology, objective, or scale lock.

### Torch 2.11 compatibility re-spike candidates

The 2026-07-20 re-spike is authorized against these exact upstream identities;
they are candidates until the repository's compatibility gates pass:

```text
upstream veRL candidate: 638b8ff84f279e054982f1f4633a546f3c6ced68
vLLM tag:                 v0.23.0
vLLM tag commit:          0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
vLLM cu129 wheel SHA256:  8bc2203995d061e6b988916b71b9dee8a5970f5fdc5f37d4445a877a2fab2cc1
TransferQueue:             0.1.8 wheel, SHA256 078c4a63ba0c222fe684e96844c937dcd97f45ac94340a9c92eb03cfbc48cffd
candidate role:           isolated compatibility re-spike only
```

Official veRL main CI/Docker material at that commit selects PyTorch 2.11 and
vLLM 0.23. Official vLLM metadata pins PyTorch 2.11, TorchVision 0.26, and
TorchAudio 2.11, and its release publishes a CUDA 12.9 x86-64 wheel but no CUDA
12.8 wheel. The first spike therefore uses the matching CUDA 12.9 PyTorch and
vLLM binaries. The existing CUDA 12.8 I8H lock remains the control; a future
CUDA 12.8 source build would be a distinct candidate and result.

The pinned veRL stable-vLLM Dockerfile is not the same resolved runtime as this
wheel candidate. At this revision it uses CUDA 13.0, Transformers 5.3, a vLLM
source checkout with unmerged vLLM PRs 44483 and 45589 applied, and an NCCL
floor needed by its suspend/resume path. This repository's candidate instead
uses the unmodified official CUDA 12.9 vLLM wheel, Transformers 4.57.6, and its
resolved NCCL 2.28.9 package. The Dockerfile is therefore upstream selection
evidence, not compatibility evidence for this exact environment. Only the
explicit `free_cache_engine=false`, `enable_sleep_mode=false`, colocated
`checkpoint_engine.backend=naive` path may be accepted by this re-spike.
Sleep/wake is unsupported unless a separately pinned runtime passes its own
gate.

Primary upstream sources:

- [veRL candidate commit](https://github.com/verl-project/verl/commit/638b8ff84f279e054982f1f4633a546f3c6ced68);
- [veRL stable-vLLM Docker candidate](https://github.com/verl-project/verl/blob/638b8ff84f279e054982f1f4633a546f3c6ced68/docker/Dockerfile.stable.vllm);
- [vLLM partial wake-up PR 44483](https://github.com/vllm-project/vllm/pull/44483);
- [vLLM reload-memory PR 45589](https://github.com/vllm-project/vllm/pull/45589);
- [veRL vLLM CI](https://github.com/verl-project/verl/blob/638b8ff84f279e054982f1f4633a546f3c6ced68/.github/workflows/vllm.yml);
- [veRL CPU CI explicit TransferQueue install](https://github.com/verl-project/verl/blob/638b8ff84f279e054982f1f4633a546f3c6ced68/.github/workflows/cpu_unit_tests.yml#L91-L94);
- [TransferQueue 0.1.8 distribution metadata](https://pypi.org/project/TransferQueue/0.1.8/);
- [vLLM v0.23.0 tag](https://github.com/vllm-project/vllm/tree/v0.23.0);
- [vLLM CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.23.0/requirements/cuda.txt);
- [vLLM CUDA installation/build contract](https://github.com/vllm-project/vllm/blob/v0.23.0/docs/getting_started/installation/gpu.cuda.inc.md);
- [PyTorch official prior-version wheels](https://pytorch.org/get-started/previous-versions/).

No vLLM or veRL source is vendored by recording these references. The
repository-owned plugin may be adapted only through public APIs and must retain
the accepted vLLM 0.12 control path until promotion is explicitly decided.

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
lifecycle state.

Pinned implementation evidence:

- [fixed review commit](https://github.com/lasgroup/SDPO/commit/7c457fc1b1f636ae794eb0362ba37d4743b06fbc);
- [root import of 1,022 files](https://github.com/lasgroup/SDPO/commit/519a257);
- [bundled veRL version](https://github.com/lasgroup/SDPO/blob/7c457fc1b1f636ae794eb0362ba37d4743b06fbc/verl/version/version);
- [Apache-2.0 license](https://github.com/lasgroup/SDPO/blob/7c457fc1b1f636ae794eb0362ba37d4743b06fbc/LICENSE);
- [teacher-context construction](https://github.com/lasgroup/SDPO/blob/7c457fc1b1f636ae794eb0362ba37d4743b06fbc/verl/trainer/ppo/ray_trainer.py#L672-L796);
- [actor teacher/loss path](https://github.com/lasgroup/SDPO/blob/7c457fc1b1f636ae794eb0362ba37d4743b06fbc/verl/workers/actor/dp_actor.py#L675-L920);
- [distillation losses](https://github.com/lasgroup/SDPO/blob/7c457fc1b1f636ae794eb0362ba37d4743b06fbc/verl/trainer/ppo/core_algos.py#L1085-L1188);
- [actor-only checkpoint manager ownership](https://github.com/lasgroup/SDPO/blob/7c457fc1b1f636ae794eb0362ba37d4743b06fbc/verl/workers/fsdp_workers.py#L893-L915).

The framework goal must include a real repository-owned SDPO reimplementation,
not merely reserve or freeze interfaces. It implements and verifies:

- feedback and successful-demonstration identity;
- a separately rendered teacher context and its token alignment;
- teacher/student token masks over multi-turn multimodal trajectories;
- teacher logits/log probabilities or a versioned approximation artifact;
- current-policy self-teacher identity plus EMA or trust-region regularization
  state;
- full-logit and top-k/tail reference-semantic loss paths;
- objective composition, metrics, teacher update, checkpoint, and resume state.

The commit above fixes the reference implementation. It does **not** yet freeze
our exact SDPO equations, teacher regularization, reprompt template, feedback
policy, full-logit/top-k choice, importance weighting, or GRPO/SDPO composition.
Those values must pass the SDPO mathematics and parity gate before an optimizer
step uses SDPO.

Upstream veRL remains the production base. The pinned SDPO repository is a
complete modified veRL `0.7.0.dev` tree whose initial commit imported 1,022
files without recording an exact upstream base SHA; its bundled
tree is therefore reference-only and is never installed, vendored, imported,
or used as a runtime fallback. The compatibility spike reimplements the
algorithm behavior on the pinned C-MAIN public surfaces without changing the
exact TGVF trajectory and observation contracts.

Known reference-implementation gaps that the spike must test explicitly:

- its distillation actor asserts that multimodal inputs are unsupported;
- its teacher reprompt path assumes a simplified text prompt/response rather
  than a complete repeated-tool trajectory;
- its on-policy worker can replace rollout behavior log probabilities with
  `log_prob.detach()`, which is forbidden here;
- it depends on veRL's legacy worker path and does not establish the new-engine
  integration required by a current upstream pin;
- its shipped path does not prove FSDP2 SDPO, LoRA-to-teacher parameter mapping,
  coexistence with a separate KL reference, or strict EMA-teacher save/resume.

Passing requires executable CPU loss/gradient parity plus exact-D multimodal
teacher replay and teacher-state checkpoint/resume parity. A config slot,
neutral interface, or static seam alone fails. Production pure-SDPO and any
SDPO+GRPO hybrid remain distinct, fail-closed objective identities: exact
mathematics and parity must be accepted before either performs a model optimizer
step.

## DeepEyes

```text
repository: https://github.com/Visual-Agent/DeepEyes
review commit: 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1
observed: 2026-07-19 JST
role: crop-call behavior, small configuration, and family-adapter design reference
dependency status: reference only; do not install or vendor
```

The crop behavior reference is the file
[`verl/workers/agent/envs/mm_process_engine/visual_toolbox_v2.py`](https://github.com/Visual-Agent/DeepEyes/blob/11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1/verl/workers/agent/envs/mm_process_engine/visual_toolbox_v2.py),
whose bytes at the review commit have SHA-256
`0d56b2ff584fe56e68f20bbb4d25a9774ecbab605ad02cdaf1dac7cd6fa8bc60`.
It establishes the public name `image_zoom_in_tool`, the `bbox_2d` argument,
clamping to source-image bounds, and cropping from the immutable original
image. Those small observable behaviors may be re-expressed behind this
project's schemas. DeepEyes-specific size/aspect heuristics, parser, prompt,
retry behavior, and rotation tool are not adopted.

Other permitted topics are limited to small configuration-composition,
family-adapter dispatch, launcher layout, and test-organization ideas that can
be re-expressed behind this project's contracts. DeepEyes code, veRL tree, and
runtime are not dependencies.

DeepEyes uses Qwen2.5-VL policy models and documents
`Qwen/Qwen2.5-72B-Instruct` as an LLM-as-judge example. The user has now fixed
our secondary policy model to `Qwen/Qwen2.5-VL-7B-Instruct` and approved
reserving—without first-pilot deployment—the 72B judge provider. Judge prompt,
reward role, and deployment topology remain unset.

Forbidden inheritance includes its observation schema or materialization,
rollout/behavior log probabilities, replay semantics, sampled-token masks,
agent loop implementation, rendered prompts, dataset assumptions, reward
coefficients, checkpoint state, and asynchronous-staleness behavior. DeepEyes
cannot supply compatibility evidence for any of those fields. This project's
native trajectories, rollout-recorded probabilities, content-addressed crop
observations, and exact TGVF observations remain authoritative.

## VLMEvalKit

```text
repository: https://github.com/open-compass/VLMEvalKit
review commit: 7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f
observed: 2026-07-20 JST
role: sole external visual-benchmark execution and official-scoring framework
dependency status: pinned external checkout plus isolated CPU/CLI runtime overlay
```

The clean detached checkout is deployed at
`/nvmesv/dredvpn009/tools/VLMEvalKit/7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f`.
Its no-dependency runtime overlay is
`/nvmesv/dredvpn009/tools/VLMEvalKit/runtime-7055d301/site-packages` and is
loaded after the checkout with the repository's `.venv312` interpreter. The
machine-readable identity and no-download validation command are documented in
`docs/VLMEVALKIT.md`. This deployment does not vendor VLMEvalKit or change the
production representation/RL dependency matrix.

Official point-in-time sources reviewed for the accepted evaluation
architecture are:

- [`run.py`](https://github.com/open-compass/VLMEvalKit/blob/7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f/run.py), SHA256 `efe24021e6f5f6ec394eba0f59afc094f897301a2ed31c6a3ba5ba975e148653`;
- [model integration documentation](https://github.com/open-compass/VLMEvalKit/blob/7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f/docs/en/Development.md), SHA256 `6572bda9bc30c7c1e870139837b649f4a7563e067d83410745b431f418288c27`;
- [configuration documentation](https://github.com/open-compass/VLMEvalKit/blob/7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f/docs/en/ConfigSystem.md), SHA256 `b1577ec11bfd2a3db91f0d25f261e55b52809bee774d3cc299126ca09a7ee006`;
- [`BaseAPI`](https://github.com/open-compass/VLMEvalKit/blob/7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f/vlmeval/api/base.py), SHA256 `ae6dc70cde9f51e2b5eea2415789c0a62a3b006323785ee71174511a26e444f9`;
- [official agent-style `extra_records` example](https://github.com/open-compass/VLMEvalKit/blob/7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f/vlmeval/api/arm_thinker.py), SHA256 `22174ccc6e7eb4703372110e7d6bf64770ebf3439f94947e723bcfceb6a64488`;
- [`image_vqa.py`](https://github.com/open-compass/VLMEvalKit/blob/7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f/vlmeval/dataset/image_vqa.py), SHA256 `c7e2eb8708d867efb10ee5645e8acd78c3ca7e0887a210125eb0143c8d5dcf26`.

The project-owned adapter follows the official `BaseAPI` boundary but owns the
complete crop/TGVF loop. It returns final answer text as `prediction` and
identity-safe trajectory metadata as `extra_records`; tensors and latent
observations remain in project-owned artifacts. Built-in `Qwen3VLChat` is
eligible only for an explicitly configured original-Qwen direct baseline. It
does not execute this project's tools, and its sampling defaults are not a
policy-evaluation configuration.

VLMEvalKit reads shared data through `LMUData`. It exposes no generic fixed-row
subset filter, and `CustomVQADataset.evaluate()` is not an available scorer.
Each source slice of a composite subset must therefore retain the official
dataset/scorer class and pass a score-parity fixture before use. The ignored,
dirty legacy checkout at `revisit_vlm/third_party/VLMEvalKit` is explicitly not
this dependency identity and must not be reused.

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
