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

## EasyR1

```text
repository: https://github.com/hiyouga/EasyR1
review commit: 07cae10c28d686a6604546617663d32e4f1089e6
observed: 2026-07-20 JST
role: Qwen3-VL decoder-only LoRA target and small Pilot configuration reference
dependency status: reference only; do not install or vendor
```

Only the exact Qwen3-VL LoRA target-module selection and the corresponding
small configuration surface may inform the Pilot implementation.  The user has
independently fixed rank 64, alpha 64, dropout zero, and an initial learning
rate of `1e-5`; those values are project decisions rather than inherited
EasyR1 defaults.  This pin does not authorize copying EasyR1's veRL fork,
trainer, rollout loop, reward code, prompts, data pipeline, or dependency lock.
Upstream veRL and the repository-owned exact-observation/replay contracts remain
authoritative.

The reviewed source is limited to the exact
[`examples/qwen3_vl_4b_geo3k_grpo_lora.sh`](https://github.com/hiyouga/EasyR1/blob/07cae10c28d686a6604546617663d32e4f1089e6/examples/qwen3_vl_4b_geo3k_grpo_lora.sh)
launcher and the LoRA stanza in
[`examples/config.yaml`](https://github.com/hiyouga/EasyR1/blob/07cae10c28d686a6604546617663d32e4f1089e6/examples/config.yaml#L43-L57).
The latter uses `all-linear` plus a visual-module exclusion. The project must
use a positive language-decoder whitelist and verify actual trainable parameter
names instead, because a name-only visual exclusion is insufficient to prove
that the merger, native DeepStack, embeddings, head, and TGVF Adapter are frozen.

## DeepEyes

```text
repository: https://github.com/Visual-Agent/DeepEyes
review commit: 11d20c6be32b2cf62c914e0c73a06db2f9a7e3a1
observed: 2026-07-19 JST
role: published data-selection method, crop-call behavior, small configuration,
  and family-adapter design reference
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

The hard data-selection reference is the accepted ICLR 2026 paper revision,
[`arXiv:2505.14362v3`](https://arxiv.org/abs/2505.14362v3), observed on
2026-07-25 JST. It specifies an eight-response difficulty gate that removes
zero- and perfect-accuracy samples, answer-format standardization and label
verification, followed by a V*-only perception-utility gate using ground-truth
regions. A recursive tree inspection of the pinned public repository commit
found no released DeepEyes-specific data-selection script. The project target
is therefore equivalent quality, properties, and source distribution under the
published method, not exact row or byte parity with an unpublished filter.
Project-specific operational choices must remain explicit and may not be
misrepresented as DeepEyes implementation details.

### Candidate source pools for Qwen3 RL selection

These are source-pinned **candidates**, not yet accepted materialized training
inputs. Download, image resolution, license completion, source-schema adapters,
and held-out leakage checks remain separate gates.

```text
V* method repository: https://github.com/penghao-wu/vstar
review commit: 4ede6647959cfb59eeabd09286adf6a5f9478da0
fine-grained candidate annotations: https://huggingface.co/datasets/craigwu/seal_vqa_data
dataset revision: 72f07263e9dd1dc5812a9ed4d8595f42cce7cf44
role: V*-style question/answer/target-instance/bbox candidate pool
license status: conditional; see inherited-source record below
```

The candidate annotation schema includes image paths, questions, answers, and
`target_instances` with `[x,y,width,height]` boxes. Images are separately
provided by COCO/GQA sources. The pinned Hugging Face annotation repository has
no dataset-card license declaration; the MIT declaration in the V* code
repository is recorded but is not treated as automatically relicensing the
separate annotation and image payloads.

Inherited image terms were checked on 2026-07-25 JST. The official V* training
instructions require COCO-2014, COCO-2017, and GQA images. The official GQA
site states that GQA images come from COCO and Flickr and that its scene graphs
derive from Visual Genome; it does not state a single replacement image
license. The COCO Consortium's terms page, pinned through website commit
`5e1c4da72464b1c6f068df0c02c91e3000ea62c4`, licenses COCO annotations under
CC-BY-4.0 but explicitly says that the Consortium does not own image copyright
and that image use must follow the applicable Flickr terms. Therefore local
research materialization is recorded with source provenance, but redistribution
or broader production promotion remains a separate legal review rather than an
implied project permission.

License/source evidence:

- [V* training-image requirements](https://github.com/penghao-wu/vstar/tree/4ede6647959cfb59eeabd09286adf6a5f9478da0#training-dataset);
- [official GQA source attribution](https://cs.stanford.edu/people/dorarad/gqa/);
- [pinned COCO terms page](https://github.com/cocodataset/cocodataset.github.io/blob/5e1c4da72464b1c6f068df0c02c91e3000ea62c4/dataset/termsofuse.htm).

The operational GQA image materialization additionally uses a pinned partial
mirror solely to avoid the official monolithic archive's transfer bottleneck:

```text
mirror dataset: https://huggingface.co/datasets/zihuwang/ReGuLaR
dataset revision: b8215201a5fd854c30135f2f0d432f032e364bfa
file: image_archives/images_gqa.tar.zst
file SHA-256: f0b15a85bb66c98cea0c7543e86706e7c92c6129660ad3f5a06879d80543caec
file bytes: 8,679,513,198
role: byte-preserving partial transport mirror for GQA images only
```

It supplies 34,847 of the 43,892 GQA images required by the pinned V*
annotations. The remaining 9,045 are fetched from the official Visual Genome
`VG_100K`/`VG_100K_2` image roots. The materialization manifest pins the mirror,
official fallback roots, per-fallback-image aggregate binding, and every final
candidate image SHA. No ReGuLaR annotation is used and the mirror does not
replace the inherited GQA/Visual Genome/Flickr terms above.

The 191-row `craigwu/vstar_bench` is evaluation data and is not the 22K-scale
pre-filter training pool.

```text
ArxivQA: https://huggingface.co/datasets/MMInstruction/ArxivQA
dataset revision: 85a6dca0e2bdc6f0268ae519be8913f83a83cafd
role: chart/scientific-figure candidate pool
license: CC-BY-SA-4.0
```

The pinned JSONL binds `id`, image path, options, question, label and rationale;
the image archive is a separate pinned payload requirement.

```text
ThinkLite-VL raw candidate: https://huggingface.co/datasets/russwang/ThinkLite-VL-70k
dataset revision: 5c86ea41d624e27e53002af47b8cf4538aa2c88f
role: general visual-reasoning candidate pool before Qwen3 difficulty selection
license: MIT as declared by the dataset card

ThinkLite-VL hard reference: https://huggingface.co/datasets/russwang/ThinkLite-VL-hard-11k
dataset revision: 541f7f463815467f80866e887e82c6e398837a08
role: downstream 11K-scale difficulty-selected reference, not the raw candidate pool
```

The raw viewer exposes image, problem, answer/ground-truth, ID and optional
choice fields. Exact file hashes and row counts must be recorded during local
materialization rather than inferred from the moving dataset name.

Other permitted topics are limited to small configuration-composition,
family-adapter dispatch, launcher layout, and test-organization ideas that can
be re-expressed behind this project's contracts. DeepEyes code, veRL tree, and
runtime are not dependencies.

The Pilot training data source is separately pinned to the public
[`ChenShawn/DeepEyes-Datasets-47k`](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k)
snapshot `5546681e28fa2eda9f60a9ea9dd0cf291216ded3` (Apache-2.0). Its complete
47K snapshot is the concatenation of these three immutable LFS objects:

| file | rows | LFS SHA-256 | bytes |
|---|---:|---|---:|
| `data_0.1.2_visual_toolbox_v2.parquet` | 22,362 | `42992bf5de25e8d766f820fb9730ece275563ba80dd41e3377bf678c9ba2c2c1` | 990,263,397 |
| `data_thinklite_reasoning_acc.parquet` | 11,031 | `660cea5ff8f74d19f993b575f30b6f5406b6c330dd8f9aacc6be59e299238967` | 1,656,152,904 |
| `data_v0.8_visual_toolbox_v2.parquet` | 13,659 | `96fc256e6f73e098c1b586f1c37baad616ecbddf1105bfca71aa07a5dda7da5a` | 2,198,504,506 |

Total row count is 47,052. The source `prompt` and its zoom-tool instructions
are forbidden inputs to the Pilot renderer. Materialization may retain only
image payload, `extra_info.question`, `reward_model.ground_truth`,
`data_source`, and the minimum typed metadata needed for verification and
provenance. The materialized manifest, row identities, shuffle seed, and
policy-prompt hash remain separate project artifacts.

DeepEyes uses Qwen2.5-VL policy models and documents
`Qwen/Qwen2.5-72B-Instruct` as an LLM-as-judge example. The user has now fixed
our secondary policy model to `Qwen/Qwen2.5-VL-7B-Instruct` and approved
the 72B model both as the sole LLM judge for VLMEvalKit benchmark evaluation
and as the fallback answer verifier for the formal Policy Pilot. The latter is
a separately identified RL-reward role: multiple choice remains rule/exact,
mathematics uses a math verifier before fallback, and open visual QA uses rules
before semantic-equivalence fallback. The RL-judge prompt, service identity,
sampling, timeout/retry policy, calibration set, and failure behavior remain
unset and must not be inherited from DeepEyes or the benchmark judge config.

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

### Qwen3-VL coordinate convention

```text
repository: https://github.com/QwenLM/Qwen3-VL
review commit: 96588727e44c78b25ba03ea03b8e12f7e64fd0da
reviewed file: cookbooks/2d_grounding.ipynb
observed: 2026-07-24 JST
role: primary-family 2D grounding coordinate convention
dependency status: reference only; do not install or vendor
```

The pinned official cookbook states that Qwen3-VL changed from Qwen2.5-VL's
resized-image absolute coordinates to relative coordinates in the range
`0..1000`. Its visualization code maps a returned Qwen3 box to original pixels
with `x * original_width / 1000` and `y * original_height / 1000`; it explicitly
says the caller need not calculate the processor-resized width. This is the
model-facing convention that a Qwen3 family adapter must reconcile with the
project's canonical immutable-source, half-open pixel box. The pin does not yet
select edge rounding or authorize an implementation change.

Pinned source:

- [Qwen3-VL 2D grounding cookbook](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/cookbooks/2d_grounding.ipynb).

### Historical Qwen3-VL Thinking comparison

```text
family: Qwen3-VL
historical size/variant: 8B Thinking
accepted stable local path:
  /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking
path presence checked: 2026-07-19 JST
full weight-directory hash: intentionally not required
```

This checkpoint was the former primary and remains the immutable identity for
completed Thinking experiments and separately identified comparisons. Decision
`QWEN3-INSTRUCT-PRIMARY-20260726` supersedes it for new representation,
policy-RL, and policy-data-selection work. Historical records retain their
exact model name, path, processor, tokenizer, template, and serialization
identities.

### Primary policy/reference and data-selection model

```text
model ID: Qwen/Qwen3-VL-8B-Instruct
official model card: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
Hugging Face revision: 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
downloaded and checked: 2026-07-26 JST
stable local path: /nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct
architecture: Qwen3VLForConditionalGeneration, BF16, no quantization
role: primary representation, policy/reference, and policy-data-selection model
```

The model was downloaded from the immutable official revision under decision
`QWEN3-INSTRUCT-TOKEN-CANARY-20260726` and promoted under decision
`QWEN3-INSTRUCT-PRIMARY-20260726`. It is a separate checkpoint from the
historical Thinking model even though the architecture and visual processor are shared.
The native Instruct template ends its generation prompt with
`<|im_start|>assistant\n`; it must not be serialized with the Thinking
checkpoint's template-owned `<think>` opener.

Downloaded weight identities are:

```text
model-00001-of-00004.safetensors
  bytes: 4902275944
  sha256: d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa
model-00002-of-00004.safetensors
  bytes: 4915962496
  sha256: 8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5
model-00003-of-00004.safetensors
  bytes: 4999831048
  sha256: 83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192
model-00004-of-00004.safetensors
  bytes: 2716270024
  sha256: 0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5
model.safetensors.index.json
  sha256: 520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070
```

Model-facing identities for the paired canary are config
`5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661`,
generation config
`8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44`,
tokenizer config
`c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399`,
tokenizer JSON
`a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7`,
preprocessor config
`27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516`,
chat-template file
`5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce`,
and effective chat-template content
`3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4`.

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
model: Qwen/Qwen2.5-72B-Instruct
Hugging Face revision: 495f39366efef23836d0cfae4fbe635880d2be31
local path: /nvmesv/dredvpn009/models/hf/Qwen2.5-72B-Instruct
benchmark service: vLLM 0.12.0, OpenAI-compatible, BF16, TP=2
role: required/fallback semantic answer verifier for fixed benchmark routes;
  optional and disabled as an RL reward for the first pilot
```

The judge is independently versioned and calibrated. It is not the frozen RL
reference policy, the SDPO self-teacher, or a replacement for executable
verifiers. The local snapshot download and service smoke are recorded in the
experiment ledger; numerical benchmark-judge calibration remains open.

## Local LaTeX layout reference

```text
file: /home/dredvpn009/Flash_Storage/projects/brian/reports/BRIAN_RC_KV_Implementation_Report.tex
file SHA256: 3c647d7decca2d95eeae635e216cb0f4a2df14a99451dade7c998e34babb7019
observed: 2026-07-21 JST
role: typography and report-organization reference only
```

The user explicitly selected this exact local file as the layout reference for
the TGVF representation-phase report. Reuse is limited to the compact A4
two-column `ctexart` presentation pattern and the high-level ordering of
method, verification, limitations, and conclusions. No BRIAN implementation,
algorithm, experiment result, prose, bibliography, or project identity is
ported into TGVF.
