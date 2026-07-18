# veRL Compatibility Spike Plan

Status: **PROPOSED — documentation-only; execution not authorized**
Drafted: **2026-07-19 JST**

The user authorized preparation of this plan. That authorization does not yet
accept the exact task, permit dependency installation, create a GPU ledger
entry, or permit a GPU process. Each later approval boundary is explicit below.

## 1. Objective and decision boundary

The spike is a bounded integration proof for the already selected upstream
veRL framework family. It is not a framework bake-off, a model-training run, or
an attempt to settle reward, data, prompt, GRPO, or SDPO research choices.

It must answer one decision:

> Can one exact upstream veRL revision support the TGVF native multi-call
> trajectory, actual behavior-policy log probabilities, immutable latent
> observation replay, both Qwen VLM family adapters, both target-condition
> provider seams, FSDP2 lifecycle, and future SDPO teacher boundaries through a
> narrow maintained/public adapter surface?

The output is either an accepted exact veRL revision/backend/extension map or a
fail-closed report. The spike may select a rollout backend and FSDP2 topology;
it may not weaken the project contracts to make a framework pass.

## 2. Macro plan and protected scope

### Objective

Produce bounded PASS/FAIL evidence and the information needed to accept or
reject the framework-binding skeleton.

### Files that an accepted spike may add or change

- disposable probe code under a separately accepted `spikes/verl_compat/`
  task boundary;
- exact golden fixtures under a separately accepted fixture path;
- `docs/VERL_COMPATIBILITY_REPORT.md`;
- one environment lock or immutable container identity per tested backend;
- complete `PLANNED`/result entries in `docs/EXPERIMENT_LEDGER.md` for GPU cells;
- the task, open-contract, and external-reference documents needed to record
  the result.

No probe package or fixture path is created merely by approving this document.

### Files and state not to touch

- no production `src/tgvf_rl/` package or framework-binding skeleton;
- no legacy repository, legacy checkpoint, or parent-directory search;
- no current Python environment mutation;
- no model-weight mutation, conversion, or full-directory hashing;
- no reward, dataset, prompt, GRPO, or SDPO default promoted into a contract;
- no 72B judge deployment;
- no SDPO or DeepEyes veRL fork installation, vendoring, submodule, or runtime
  import.

### Unresolved decisions that remain unresolved after approving the plan

- production veRL revision and rollout backend;
- local/runtime path for `Qwen/Qwen2.5-VL-7B-Instruct`;
- exact FSDP2 topology beyond the minimum probe;
- policy LoRA versus full-parameter training scope;
- exact GRPO and SDPO equations;
- production data, reward, prompt, and tool-call cap.

## 3. Candidate upstream revisions

Only official `verl-project/verl` revisions are candidates.

### C-MAIN — primary feature candidate

```text
repository: https://github.com/verl-project/verl
branch snapshot: main observed 2026-07-19 JST
commit: e003163181731412595257a72ec173071efb125f
commit date: 2026-07-17
reported package version: 0.9.0.dev
role: primary exact spike candidate
dependency status: review identity only; not installed or pinned for production
```

C-MAIN is tested first because it contains relevant post-v0.8.0 determinism work
and the current Qwen3 tool-call/token-preservation fixes, as well as FSDP2,
Qwen3-VL and Qwen2.5-VL examples, multi-turn AgentLoop support, rollout
`response_logprobs`, extensible AgentLoop `extra_fields`, and multimodal
on-policy-distillation paths. It is an exact development snapshot rather than
a stable release; that risk is recorded and none of these features alone proves
the TGVF contract.

Static review has already identified three material risks:

1. the released `ToolResponse` schema carries only text, image, and video, so a
   native latent main-`D`/D-DeepStack response is not a built-in tool payload;
2. the default AgentLoop multimodal post-processing decodes token IDs and calls
   the processor again to rebuild multimodal inputs, which cannot be used for
   exact TGVF replay;
3. the stock Qwen3-VL training forward starts from `pixel_values` and runs the
   visual module again to create image and DeepStack embeddings, rather than
   consuming the exact rollout-materialized latent bundle.

The spike must therefore prove a narrow public override/adapter path. Reusing
those default paths and calling the result latent replay is a hard failure.

Official C-MAIN full-determinism support also has a strict boundary: vLLM
single-turn rollout is supported, while SGLang and multi-turn
`tool_agent_loop` are not bitwise reproducible. Whole asynchronous multi-turn
rollout bitwise equality is diagnostic, not a substitute for this project's
actual contract. The hard requirements are actual sampled behavior log
probabilities plus deterministic actor/reference/teacher forward replay on the
fixed materialized trajectory. Deterministic-operation warnings are promoted to
test failures.

### C-STABLE — stable comparison

```text
repository: https://github.com/verl-project/verl
release: v0.8.0
commit: 7aed6b230776f963fa09509c10d9c3a767d1102c
released: 2026-06-01
role: latest stable comparison and fallback only if it meets every hard gate
dependency status: review identity only; not installed or pinned for production
```

C-STABLE establishes how much of the required surface exists in the latest
stable release. It does not replace C-MAIN automatically: rollout-level full-determinism
support landed after v0.8.0, and static comparison must enumerate all other
relevant differences across the roughly 200 intervening commits. C-STABLE is run on
GPU only if S0 shows that it can meet every hard gate through maintained/public
interfaces. Selecting C-MAIN requires an explicit stability/maintenance
rationale; selecting C-STABLE requires proof that no needed C-MAIN correctness
fix is lost.

The following are not candidates:

- the veRL tree embedded in `lasgroup/SDPO`;
- the veRL tree or dependency environment embedded in DeepEyes;
- an unrecorded moving branch or unversioned package install;
- a local fork of private trainer internals.

## 4. Candidate dependency matrices

These are upstream comparison baselines, not approved local locks. Before any
installation, the host driver/runtime and container support must be inventoried
read-only, an isolated path or immutable image digest must be approved, and the
resolved package graph must be recorded.

| ID | veRL source | Rollout backend baseline | Upstream environment baseline | Status |
|---|---|---|---|---|
| `M-VLLM-MAIN` | C-MAIN exact checkout | current Docker argument vLLM `0.23.0` | Python `3.12`, CUDA `13.0.2`, PyTorch `2.11.0`, Transformers `5.3.0`, qwen-vl-utils `0.0.14`, FlashAttention `2.8.3` | proposed primary |
| `M-SGLANG-MAIN` | C-MAIN exact checkout | `lmsysorg/sglang:v0.5.12` | CUDA `13.0.2`, Transformers `5.3.0`, qwen-vl-utils `0.0.14`; image-owned Python/PyTorch and image digest must be captured | proposed functional comparison |
| `M-VLLM-080` | C-STABLE exact checkout | release baseline vLLM `0.20.2` | exact C-STABLE recipe, source checkout, and resolved graph | stable comparison |
| `M-SGLANG-080` | C-STABLE exact checkout | release baseline SGLang `0.5.12` | exact C-STABLE recipe, source checkout, and resolved graph | stable comparison |

The official Dockerfiles cannot serve as locks. At C-MAIN, the vLLM file's comment
still says `0.20.2` while its argument says `0.23.0`, and both vLLM/SGLang files
still set `VERL_VERSION=v0.7.1`. C-STABLE recipes have a similar source-version
mismatch. Any approved environment must load the exact candidate checkout,
verify the runtime veRL commit/version, capture the image digest and resolved
package graph, and record these upstream recipe discrepancies.

vLLM is the first full correctness candidate because C-MAIN documents rollout
determinism only there. SGLang receives a bounded functional/logprob/latent-
transport smoke only when S0 finds a plausible public adapter surface. It cannot
be selected for the first pilot until it independently closes deterministic
forward/replay and sampling-identity gates. The spike selects one production
backend; it does not create a two-backend maintenance obligation. Correctness
and extension-surface stability outrank throughput.

### 4.1 Model support levels

Compatibility is reported by level rather than a single misleading boolean:

| Level | Claim |
|---|---|
| `L0` | pinned tokenizer/processor/native-template identity, zero tokenizer growth, parser, and exact transcript fixtures |
| `L1` | deterministic base VLM source-image forward/replay behind the family adapter |
| `L2` | synthetic main-`D` injection, immutable transport, and replay without visual recomputation |
| `L3` | repeated calls with main `D`, every model-supported DeepStack branch, positions, masks, and cache parity |
| `L4` | paired single-device/FSDP2 update, checkpoint, teardown, resume, and next-step parity |

Qwen3-VL-8B-Thinking must pass `L0..L4` for the selected stack. The initial
Qwen2.5-VL skeleton claim requires `L0..L1`; `L2..L4` are diagnostic/deferred in
this spike and must pass later, with a separate family-specific representation
artifact, before any Qwen2.5 end-to-end TGVF claim. A missing DeepStack-
equivalent path is reported at `L3`, never hidden by a dummy branch.

### 4.2 Cell matrix

The selectable stack identity is the tuple `(veRL commit, rollout backend,
resolved environment, repository adapter surface)`. A backend comparison can
fail without invalidating an independent mandatory cell.

| Cell | Revision/backend | Model/level | Stage | Class | Blocks A4? | Prerequisite | Maximum GPU cost |
|---|---|---|---|---|---|---|---|
| `SC-00` | C-MAIN / vLLM | source/API ownership | S0 | mandatory | yes | A0 | 0 |
| `SC-01` | C-STABLE / vLLM | exact delta from C-MAIN | S0 | diagnostic | no | A0 | 0 |
| `SC-10` | backend-neutral | Qwen3 `L0` | S1 | mandatory | yes | A1 | 0 |
| `SC-11` | backend-neutral | Qwen2.5 `L0` | S1 | mandatory | yes | A1 + accepted processor/template snapshot | 0 |
| `SC-20` | C-MAIN / vLLM | Qwen3 `L1..L3` | S2 | mandatory | yes | A2 ledger | 0.75 GPU-hour |
| `SC-21` | C-MAIN / SGLang | Qwen3 `L1..L3` | S2 | diagnostic | no | S0 public seam + A2 | 0.75 GPU-hour |
| `SC-22` | C-MAIN / vLLM | Qwen2.5 `L1` | S2 | mandatory | yes | accepted model runtime + A2 | 0.75 GPU-hour |
| `SC-23` | C-MAIN / vLLM | Qwen2.5 `L2..L3` | S2 | diagnostic/deferred | no | SC-22 + A2 | 0.75 GPU-hour |
| `SC-30` | C-MAIN / vLLM | Qwen3 `L4`, single-device control plus two-rank FSDP2 | S3 | mandatory | yes | SC-20 + A2 | 3 GPU-hours |
| `SC-40` | C-MAIN / backend-neutral | SDPO/OPD static seam and state schema | S0/S1 | mandatory | yes | A0/A1 | 0 |
| `SC-41` | selected stack | teacher forward/state round trip | S4 | diagnostic/deferred | no | separate A2 | 1 GPU-hour |
| `SC-50` | C-STABLE / vLLM | Qwen3 `L1..L4` | S2/S3 | conditional | only if selecting C-STABLE | S0 admission + separate A2 | separately approved |

`SC-11`/`SC-22` are mandatory because the accepted project task requires a
real Qwen2.5 family-adapter fixture in the initial skeleton. If its runtime
identity is not approved, the overall result is `INCOMPLETE`, not a failure of
veRL. Its later `L2..L4` end-to-end claim does not block a Qwen3 framework pin.
SGLang and C-STABLE cells do not block acceptance of C-MAIN/vLLM unless the
report proposes selecting them. Selecting C-STABLE requires rerunning all
applicable mandatory cells on C-STABLE; a single failed-cell rerun is
insufficient.

## 5. Fixed spike fixtures

The fixtures are deliberately tiny and deterministic. They exercise framework
contracts without requiring a trained native TGVF Adapter.

### F0 — native transcript fixtures

For each model family:

- direct response with no tool call;
- one valid `tgvf_focus_tool` call and continuation;
- two sequential `tgvf_focus_tool` calls and continuation;
- malformed/unknown tool calls for fail-closed parser behavior;
- exact token IDs, chat-template hash, tool-schema hash, stop IDs, and ownership
  masks;
- no project-specific token and no tokenizer-size change.

Qwen3-VL-8B-Thinking has exactly one native opening `<think>` per assistant
reasoning turn, with its template-versus-policy ownership frozen by the fixture.
Qwen2.5-VL-Instruct follows its own pinned native template and is not given a
Qwen3 thinking prefill. If its fixed snapshot cannot express the required
native multimodal tool transcript, the family cell fails or remains blocked;
injecting a Qwen3/DeepEyes custom template cannot make it pass.

The model's accepted native processor/chat template owns serialization. The
DeepEyes custom template and any convenience rerenderer are not fixtures.

The two-call structural fixture is scripted or token-constrained so framework
transport does not depend on the untrained base policy spontaneously choosing
two calls. Any constraining logit processor is fully versioned in F3, and this
fixture does not measure natural tool use. A separate nontrivial sampled-token
fixture validates actual behavior-logprob semantics without requiring two
spontaneous calls.

### F1 — synthetic latent observation bundle

Each tool call materializes a deterministic, lossless bundle keyed by an
`observation_id`. Tensor dimensions are derived from the tested family config;
values are generated from a recorded fixture seed. The bundle includes:

```text
schema/version and observation_id
call index and target-span/token identity
rollout policy and representation-fixture version
condition provider/version and exact Hq or target-embedding provenance artifact
condition input shape, dtype, content checksum, and target-span checksum
main D tensor
every model-supported D-DeepStack branch
shape, dtype, stride/layout, storage/checksum identity
native visual span and grid/layout identity
M-RoPE/position IDs and rope deltas
multimodal token types
attention, visibility, source-image, and D masks
cache append/reset/reuse contract
```

The condition input is retained as representation-materialization provenance;
likelihood/reference/teacher replay consumes the recorded `D` bundle and never
consumes or regenerates `Hq`. Rollout-time contextual-provider extraction from
the same actor is allowed only before the bundle is materialized and versioned.

“Storage identity” means a content-addressed blob and alias graph, not a Python
object ID, device pointer, or process-local storage address. Each cache key
includes policy version, observation ID, and turn/call index. Cached
continuation logits/logprobs must match a canonical no-cache full-prefix forward
within the declared tolerance; current, proximal-old, reference, and teacher
workers never share an unowned KV cache.

The tool transcript contains only native Qwen tokens. The tensor bundle travels
out of band by value or immutable content-addressed handle. PIL conversion,
image codec round trips, processor re-encoding, lossy serialization, and a
proximal-old/current/reference/teacher call back into the TGVF Adapter
are forbidden.

For Qwen3-VL, the fixture must exercise main `D` and every required DeepStack
branch. For Qwen2.5-VL, the family adapter must report its actual supported
branch contract. Absence of a valid DeepStack-equivalent path blocks an
end-to-end Qwen2.5 support claim; it must not be hidden with zero/dummy tensors.
The spike may still pass the narrower processor/transcript/base-forward adapter
cell if that limitation is explicit.

### F2 — target-condition provider seams

The same observation fixture is produced once from a deterministic contextual
hidden-state input and once from a deterministic target-token-embedding input.
This proves schema and dispatch compatibility only. It does not claim that
either provider is scientifically adequate or that a native representation
checkpoint exists.

### F3 — sampling identity

Each generated assistant token records:

```text
token ID and assistant-turn/call index
policy-sampled ownership mask
actual behavior log probability
raw-model log probability when available, under a different field name
whether each value is before or after every sampling transform
temperature, top-p, top-k, min-p, penalties, processors and their order
stop/EOS ownership
backend and backend version
policy/weight-sync version
global/per-sample seed and RNG derivation/state
worker/rank identity and asynchronous staleness
```

The probe explicitly enables veRL rollout log-probability calculation and uses
a nontrivial sampling configuration so transform semantics are observable. For
the C-MAIN vLLM path, the spike must verify whether the returned
`processed_logprobs` are the normalized probability of the actual post-
transform sampling distribution; the field name alone is not evidence. Any
raw/pre-transform value is preserved separately.

The behavior value used by a future ratio is defined as the log probability of
the sampled token under the final normalized distribution after temperature,
all penalties/processors, top-k, top-p, min-p, and every other enabled sampling
transform in exact execution order. S0 traces that dataflow for each backend.
S1 uses a tiny fixed-logits oracle with temperature not equal to one, finite
top-k, top-p below one, and at least one stateful penalty/processor to calculate
the truncated and renormalized distribution independently. Stop strings, EOS,
tool-call terminators, and any backend-stripped token have explicit sampled/
returned/owned/logprob semantics. Raw-model logprob is diagnostic only and can
never be selected as behavior logprob by configuration.

The first proof has staleness zero. No policy update may occur between a sample
and the replay that consumes its behavior log probabilities.

## 6. Staged execution

Each cell stops at its first hard failure and preserves evidence. Independent
comparison cells may continue; a dependent cell cannot waive or bypass an
earlier failure.

### S0 — static source and ownership map

Authorization required: acceptance of this plan only.

Work:

- confirm candidate commits/tags and licenses;
- map documented extension points for AgentLoop, tools, rollout output,
  DataProto/TensorDict, custom model forward, policy loss, FSDP2, checkpoint,
  and teacher/distillation lifecycle;
- inspect Qwen-VL monkey patches, `external_lib`, custom AgentLoop manager,
  rollout-server/replica, training-engine, and model-type registration rather
  than depending on an accidental Hugging Face subclass;
- identify every private/experimental method that an apparent implementation
  would otherwise need to replace;
- trace the stock Qwen3/Qwen2.5 visual-forward paths and prove where a recorded
  main-`D`/D-DeepStack bundle can bypass visual recomputation;
- compare C-MAIN and C-STABLE determinism support, including the documented vLLM
  single-turn, SGLang, and multi-turn limitations;
- trace both rollout-logprob use and default old-logprob recomputation so the
  later objective can select an approved behavior/proximal contract without
  inheriting a trainer mode;
- map public ownership for group construction, group mean/std, advantage
  scaling, behavior/proximal ratio, clipping, reference/KL, token masks,
  token/sequence normalization, microbatch/global denominators, and gradient
  accumulation; a replaceable scalar loss alone is insufficient;
- map the pinned SDPO reference patch surface to C-MAIN public hooks and record
  any C-STABLE surface difference;
- select the intended non-deprecated sync trainer entry for the probe; success
  through a legacy `main_ppo` example alone does not pass;
- resolve the proposed dependency matrices against the host without installing
  them.

Output: static API/ownership table and dependency-install proposal.

### S1 — isolated CPU/schema probe

Authorization required: separate dependency/environment approval after S0.

Work:

- load only tokenizer/processor/config artifacts for both fixed model IDs;
- verify zero tokenizer growth and exact F0 serialization round trips;
- pass F1/F2/F3 records through batching, grouping, Ray serialization, and
  DataProto/TensorDict boundaries without model execution;
- verify exact checksums after single, batched, reordered, and two-call paths;
- verify absent fields, shape mismatches, family mismatch, stale handles, and
  checksum mismatch all fail closed.
- run the F3 fixed-logits post-transform probability oracle;
- pass distinct sentinel values for every future GRPO-owned field through the
  public dataflow and prove that framework preprocessing neither overwrites nor
  silently normalizes them.

No optimizer step and no GRPO/SDPO numerical fixture occurs in S1.

### S2 — single-device rollout and replay probe

Authorization required: a complete per-cell `PLANNED` GPU ledger entry.

Work for C-MAIN with vLLM first, a bounded SGLang functional cell, and C-STABLE
only when S0 admits the stable comparison:

- primary model: the accepted local Qwen3-VL-8B-Thinking path, not the upstream
  Qwen3-VL-8B-Instruct example;
- secondary fixture: `Qwen/Qwen2.5-VL-7B-Instruct` after its exact local/runtime
  identity and any weight acquisition are separately approved;
- direct, one-call, and two-call F0 trajectories;
- actual F3 behavior log probabilities for all and only policy-sampled tokens;
- explicit `calculate_log_probs=true` and a nontrivial transform configuration,
  with no reliance on a backend default;
- exact F1 latent injection during rollout, followed by proximal-old when the
  later objective requires it, current-policy, and frozen-reference replay of
  the same recorded bundle;
- single versus batched and both F2 provider seams;
- no decode/rerender/retokenize drift and no hidden processor pass over `D`;
- no tool-observation `D` span takes a stock
  `pixel_values -> visual module -> DeepStack` recomputation path on rollout,
  current, proximal-old, reference, or teacher replay; the original source
  image follows a separate frozen deterministic vision contract;
- deterministic-operation warnings captured and upgraded to cell failures;
- fixed recorded trajectory/observation replay satisfies deterministic forward
  parity with dropout zero and staleness zero. Repeating an asynchronous
  multi-turn rollout in a separate run is diagnostic and need not be bitwise
  identical;
- measured peak memory, wall time, tool latency, and replay throughput.

No policy update occurs in S2.

### S3 — minimum FSDP2 lifecycle probe

Authorization required: a separate complete `PLANNED` entry and at least two
approved GPUs.

Work:

- a paired single-device control and two-rank FSDP2 actor with the exact same
  initialization, ordered batch, probe objective, optimizer, dtype, and seed;
- separate frozen reference identity restored on both paths;
- LoRA-first policy scope with dropout zero; exact module whitelist recorded;
- load, deterministic forward/backward, one infrastructure-only update, sharded
  save, teardown, resume, and one matching next step;
- exact optimizer, scheduler, RNG, sampler, policy-version, and weight-sync
  state restoration;
- synchronous step-boundary checkpointing with asynchronous save disabled;
- explicit round trip of dataloader position and every project-owned
  observation/objective/teacher schema state rather than assuming the stock
  checkpoint manager owns them;
- one direct and one two-call F1 replay batch;
- compare pre-update loss, per-module gradient direction and magnitude, LoRA
  parameter delta, post-update loss, and the uninterrupted/resumed next step;
- record actor/reference device meshes, sharding/wrap policy, offload, mixed
  precision, and state-dict ownership;
- state-dict strategy and eventual HF/PEFT export boundary recorded.

The scalar objective is a deterministic non-RL probe objective declared in the
ledger. It may validate FSDP2 mechanics but may not be called GRPO. No GRPO
optimizer step is allowed until the exact project equations are accepted.
Full-parameter FSDP2 remains a required capability and gets its own execution
gate before the first full-parameter experiment; it is not silently inferred
from this LoRA probe.

### S4 — SDPO readiness seam

Authorization required: S1 for schema work; a complete GPU ledger entry for any
teacher forward.

Work:

- map the pinned
  `lasgroup/SDPO@7c457fc1b1f636ae794eb0362ba37d4743b06fbc`
  dataflow/loss/teacher state to C-MAIN and the admitted C-STABLE comparison;
- serialize a complete native two-call teacher context rather than a simplified
  final response;
- align teacher targets to every policy-sampled assistant token across turns;
- pass the same F1 observation handles to student, frozen reference, and
  self-teacher paths while keeping behavior, proximal-old, current, frozen
  reference, and self-teacher identities distinct;
- reserve full-logit and versioned top-k/tail target schemas and measure their
  memory/I/O bounds;
- require the teacher target mask to equal the policy-sampled assistant-token
  mask; template, tool-response, image, and padding tokens may be context but
  never distillation targets;
- map the official veRL OPD teacher/top-k/multimodal/FSDP hooks before adding an
  SDPO-specific boundary;
- define current-policy/EMA teacher lifecycle, FSDP2 ownership, checkpoint,
  RNG, and strict resume fields;
- verify that the optional 72B answer judge remains a separate disabled
  provider.

This stage produces `SDPO_STATIC_SEAM_READY`, not runtime SDPO compatibility.
`SC-41` may additionally test a teacher forward/state round trip, but no SDPO
loss or optimizer step is allowed until its equations and parity oracle are
accepted.

### S5 — decision report

Produce `docs/VERL_COMPATIBILITY_REPORT.md` with:

- result for every cell and links to immutable logs/fixtures;
- selected veRL commit, backend, dependency lock/image digest, and public
  extension surface;
- rejected alternatives and exact failure evidence;
- FSDP2 topology/state-dict evidence and measured resources;
- actual behavior-logprob semantics;
- exact observation/replay schema and parity results;
- Qwen3 and Qwen2.5 support level stated separately;
- SDPO patch-surface and teacher-lifecycle map;
- all unresolved blockers and a recommendation to accept or reject the
  framework-binding skeleton.

Only an explicitly accepted S5 report becomes a production veRL pin.

## 7. PASS/FAIL contract

### 7.1 Structural equality

The following are exact, not approximate:

- tokenizer vocabulary size, token IDs, tool schema, call order, and token-
  ownership masks;
- canonical bytes for template-owned prompt/environment construction;
- original sampled assistant token IDs/bytes exactly as generated. Parsing a
  tool call never authorizes canonical JSON rendering, message reconstruction,
  or replacement of sampled history;
- observation IDs, call indices, shapes, dtypes, layouts, branch order, position
  IDs, token types, masks, cache contract, and content checksums;
- policy/backend/sampling identities and the count/alignment of recorded
  behavior log probabilities;
- non-floating checkpoint metadata, optimizer step, scheduler step, policy
  version, sampler/RNG identity, and resume cursor.

Every F1 tensor must be bit-preserving across transport and save/load. A content
handle must resolve to the same checksum on rollout, proximal-old when present,
current-policy, reference, and teacher workers.

### 7.2 Proposed numerical tolerances

These tolerances are part of the plan acceptance; they are not library
defaults.

| Comparison | Proposed requirement |
|---|---|
| deterministic FP32 unit replay, full logits | paired sampled-token mask; `atol <= 1e-6`, `rtol <= 1e-5` |
| BF16 same-policy rollout/replay | identical token IDs; sampled-token logprob and fixed top-16 logits max absolute delta `<= 5e-3`; no non-finite value |
| latent path replay | same absolute FP32/BF16 caps as above; latent-versus-direct error ratio is diagnostic only |
| actual transformed behavior logprob versus independent reconstruction | FP32 `atol <= 1e-4`; BF16 max absolute delta `<= 5e-3` |
| two-rank FSDP2 versus single-device probe loss | `atol <= 1e-4`, `rtol <= 1e-3` |
| nonzero trainable gradients | per-module cosine `>= 0.999`, norm ratio in `[0.99, 1.01]`, and elementwise max delta `<= 1e-4` FP32 or `<= 5e-3` BF16; frozen parameters have no gradient |
| LoRA parameter-update delta | paired elementwise max delta uses the gradient dtype cap above; update norm ratio in `[0.99, 1.01]` |
| resumed next step | loss/logprob/logits/gradients/parameter delta meet the paired caps; optimizer slots, scheduler, policy version, RNG and sampler cursor are structurally exact |

The direct/no-tool control is measured before latent injection. If it exceeds
an absolute hard cap, the cell fails; the cap is not relaxed to accommodate a
noisy backend. Relative latent/direct multipliers are metrics only because the
direct error can be near zero. Cross-backend equality is diagnostic, not
required: each backend must independently satisfy replay against its own
actual behavior distribution. Every reduction reports mask, dtype, token count,
aggregation, and denominator.

### 7.3 Hard PASS requirements

An overall PASS requires all applicable items:

1. C-MAIN, or an explicitly justified exact C-STABLE revision, is reproducible
   in an isolated environment.
2. Native direct/one-call/two-call transcripts round-trip exactly with no
   tokenizer growth or sampled/template ownership ambiguity. Qwen3 has no
   duplicate `<think>` opener; Qwen2.5 follows its own pinned native template.
3. Every policy-sampled token in every assistant turn has a finite actual
   behavior log probability whose pre/post-transform meaning is proven. A raw
   model probability is stored separately and cannot masquerade as behavior
   probability; rollout log-probability collection is explicitly enabled.
4. Main `D`, every required D-DeepStack tensor, visual layout, positions,
   multimodal types, masks, and cache contract remain bit-preserving and are
   consumed unchanged by rollout/proximal-old/current/reference/teacher replay
   as applicable.
5. No likelihood/reference/teacher replay calls the TGVF Adapter,
   decodes `D`, converts it to PIL, or reprocesses the tool-observation span
   through a vision encoder. The original image remains a separate fixed input.
6. Same-version deterministic replay satisfies the numerical table for direct,
   one-call, two-call, single, and batched fixtures with staleness zero.
7. Qwen3-VL-8B-Thinking passes `L0..L4`. Qwen2.5-VL passes mandatory `L0..L1`
   and reports `L2..L4` separately without a false end-to-end claim.
8. Both target-condition provider seams produce the same accepted observation
   schema and pass batching/replay.
9. FSDP2 completes load/update/save/teardown/resume with strict state continuity
   through a maintained engine/checkpoint surface.
10. Public seams exist for every GRPO-owned dataflow named in S0, and sentinel
    tests show no silent overwrite or normalization. No private trainer fork is
    required; exact equations and optimizer parity remain an A3/Gate-G0 task.
11. Complete multi-call multimodal SDPO teacher context, alignment, masks,
    targets, teacher/reference separation, FSDP2 ownership and strict-resume
    fields fit the initial skeleton boundary without using the reference fork as
    runtime code. The result is `SDPO_STATIC_SEAM_READY`, not runtime SDPO.

### 7.4 Hard FAIL conditions

Any applicable condition fails the candidate:

- modifying or replacing private veRL trainer/worker internals is required;
- the only tool path is text/PIL/video or latent tensors are processor/codec
  re-encoded;
- `D`, layout, positions, masks, branches, or cache state are recomputed by
  proximal-old/current/reference/teacher models, or replay regenerates the
  rollout-time `Hq` provenance artifact;
- actual sampling probabilities are unavailable, ambiguous, missing for any
  sampled assistant token, or replaced by `new_logprobs.detach()`;
- the spike silently assumes `pi_old = pi_rollout` or chooses a veRL rollout-
  correction mode before the project's exact objective equations are accepted;
- sampled history is decoded/rerendered/retokenized with any token drift;
- parsed assistant JSON is canonicalized and substituted for the original
  sampled assistant tokens;
- a policy update intervenes before behavior-logprob replay, or asynchronous
  staleness is unbounded/unrecorded;
- tokenizer growth, custom project tokens, duplicate `<think>`, or loss-mask
  ownership overlap occurs;
- FSDP2 cannot strictly save/resume policy, optimizer, RNG, sampling, and
  version state;
- SDPO teacher replay is text-only, final-turn-only, or conflates self-teacher,
  frozen RL reference, and answer judge;
- family-specific Qwen tensors leak into shared objective/trajectory code or an
  unsupported branch is silently zero-filled;
- cached and no-cache full-prefix replay exceed tolerance, or a KV cache is
  reused across an incompatible policy/observation/turn identity;
- the environment cannot be reproduced from an exact source commit and package
  or image identities.

A narrow repository-owned adapter is allowed only when it subclasses or calls a
documented extension point and remains outside veRL private control flow. The
report must enumerate that adapter's files and maintenance budget.

## 8. Bounded resources

These are maximums, not reservations.

| Stage | CPU/RAM | GPU | Wall-time cap | Other cap |
|---|---|---|---|---|
| S0 | ordinary source review | 0 | 4 hours | no install |
| S1 | up to 16 CPU cores / 64 GiB RAM | 0 | 4 hours per matrix | no model-weight load |
| S2 | up to 16 CPU cores / 64 GiB RAM | 1 approved GPU per cell | 45 minutes per model/backend cell | at most 8 trajectories and 256 sampled tokens per turn; scripted structural tokens are separate |
| S3 | up to 32 CPU cores / 128 GiB RAM | at least 2 approved GPUs for the distributed half | 90 minutes aggregate across paired cells; each launch `<= 60` minutes | at most 2 logical optimizer steps per paired path |
| S4 | S1 resources for schema; S2 resources for a teacher forward | 0 or 1 approved GPU | 60 minutes | no SDPO optimizer step |

Global proposed GPU cap: **10 GPU-hours**, including reruns and a 25% failure
contingency for C-MAIN mandatory cells and the listed optional cells, as
itemized in §4.2. A full C-STABLE qualification is outside that cap and requires
a revised resource approval. No individual process may run longer than 60
minutes without a new ledger entry and approval. Stop a cell immediately after
a hard failure is captured.

Proposed scratch capacity is at most **150 GiB excluding existing model
weights** for two isolated backend environments, locks, and logs. Any new model
download, container pull, or additional cache allocation is separately listed
before approval. The Qwen2.5 cell remains `BLOCKED_NOT_RUN`, not silently
skipped or treated as a framework failure, until its runtime identity is
accepted.

Exact GPU model/count, host driver, CUDA compatibility, environment path/image
digest, output path, commands, and start/stop conditions remain `[TBD]` until
the corresponding `PLANNED` ledger entry. Four-GPU or long-running training is
outside this spike.

## 9. Result vocabulary

Each cell has exactly one result:

- `PASS`: all applicable evidence and tolerances satisfied;
- `FAIL`: a hard condition was violated, with evidence;
- `BLOCKED_NOT_RUN`: authorization or a declared external input was absent;
- `INVALID`: the environment or fixture identity was wrong, so the result says
  nothing about compatibility.

The overall recommendation is one of:

- `ACCEPT_STACK(commit, backend, environment, adapter_surface)`, with either
  C-MAIN's maintenance rationale or C-STABLE's full mandatory-cell rerun;
- `REJECT_CURRENT_VERL_STACK`;
- `INCOMPLETE` when a mandatory cell is `BLOCKED_NOT_RUN` or `INVALID`.

There is no overall “pass with an unresolved hard blocker.”

## 10. Approval checkpoints

These approvals implement
[`OPEN_IMPLEMENTATION_CONTRACTS.md` Gate V0](OPEN_IMPLEMENTATION_CONTRACTS.md#6-gate-v0--before-the-verl-compatibility-spike).
Every accepted row records the repository commit containing this plan, date,
accepted scope, approver, and evidence/ledger IDs.

| Approval | Current status | Date | Accepted scope | Evidence |
|---|---|---|---|---|
| A0 plan | `PROPOSED_NOT_ACCEPTED` | — | — | this draft only |
| A1 environment + S1 | `NOT_ACCEPTED` | — | — | — |
| A2 GPU cell | `NOT_ACCEPTED` | — | no cell | — |
| A3 objective | `NOT_ACCEPTED` | — | neither GRPO nor SDPO | — |
| A4 report/pin | `NOT_ACCEPTED` | — | — | — |

1. **A0 — plan acceptance:** the user accepts this exact task, matrix,
   fixtures, tolerances, failure rules, and resource ceilings at a recorded git
   revision. This closes only `VS-01` through the plan-level portions of
   `VS-07`; it authorizes S0, not installation or GPU work.
2. **A1 — dependency approval:** after S0, the user accepts exact isolated
   environment path/image digests, resolved package locks, download/storage
   requirements, and install commands. It authorizes materializing only those
   environments and running S1.
3. **A2 — GPU-cell approval:** before each S2/S3/S4 GPU cell, the user accepts a
   complete `PLANNED` ledger entry containing its `SC-*` ID, exact hardware,
   commands, outputs, and limits. One A2 does not authorize another cell.
4. **A3 — objective approval:** exact GRPO equations are accepted before any
   GRPO optimizer fixture; exact SDPO equations receive a separate later
   approval before any SDPO optimizer fixture.
5. **A4 — report/pin acceptance:** the user accepts S5 evidence and the exact
   stack tuple. This permits proposing a separate framework-binding skeleton
   task; it neither creates that code nor automatically closes `SK-01..SK-12`.

Until A0 is explicitly recorded, this file is a proposal and no spike code is
authorized.
