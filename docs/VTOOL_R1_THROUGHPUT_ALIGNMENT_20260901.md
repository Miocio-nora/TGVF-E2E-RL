# VTool-R1 时间优化前七项：源码对齐与验证边界（2026-09-01）

## 当前结论

本轮先在统一的 `policy-e2e-method-matrix-run-config-v2` 接口下实现前七项时间优化的源码路径和 CPU 合约验证面。它们不是七个散落的命令行补丁，也没有把 @512、1M 或 `2048²` 写进 runtime；七项中的配置选择都从 TOML 进入 typed config、veRL override 和运行 receipt。在此基础上又补强了三条 runtime 热路径：KL=0 时关闭非目标 reference diagnostic、checkpoint step 复用单次权重 publication，以及 exact replay decoder batching。这三条是执行 invariant/能力，不伪装成可随意切换的实验参数。

当前状态只应表述为：**source/CPU implemented，真实 Ray/GPU throughput canary 未运行**。因此本文件不提供“达到 VTool-R1 速度”“一天完成 50 step”或任何 B200/H100 加速倍数，也不登记训练结果。旧的 method-matrix v1 配置继续按历史默认读取；本轮没有修改 canonical 训练配置、没有启动训练或评测。

这里的“对齐”表示参考 [VTool-R1 training-v2](https://github.com/VTool-R1/training-v2) 的公开实现选择，并将对应能力接入本项目的严格 action boundary、multimodal exact replay、TGVF/Atomic observation 和 reward identity 链路；它不表示两个项目的模型、数据、分辨率、reward、replay 或硬件完全相同。

## 七项实现及边界

| # | 优化 | 本项目实现 | 必须保留的边界 |
|---:|---|---|---|
| 1 | Native async trajectory | framework-neutral loop 以可恢复的 sampling boundary 驱动；veRL owner event loop 直接 await policy server turn。解析、工具执行、observation/replay 构造等同步段只在边界间进入 worker，不再为整条 trajectory 停驻一个线程；取消会先排空已开始的副作用段，再释放 sticky trajectory。 | 这是并发结构的源码/CPU 验证，尚未证明 Ray 下的吞吐、取消恢复或长时稳定性。sync 兼容路径仍保留，但不是 v2 的预期热路径。 |
| 2 | Dynamic token batching | actor、rollout logprob，以及实际启用时的 reference logprob 分别绑定 dynamic-bsz 和各自 token cap；固定 micro-batch size 在该路径置空。actor loss 使用独立的 `tgvf_dynamic_global_token_mean` identity，对全局 policy-token 数归一化，避免可变 micro 按“每个 micro 等权”产生 packing-dependent 梯度。exact replay 现已在每个 microbatch 内独立校验/重建各 trajectory，再把变长 injected inputs 右侧补齐，用一次 Qwen decoder forward 处理整个 microbatch。 | **不能沿用 DeepEyes 固定 equal-micro reduction，也不能称为纯 wall-clock 开关。** global policy-token mean 相对 fixed-control 的 equal-micro reduction 是可审计的 reduction/objective 变化；正式速度或效果比较必须单列这一 intervention。当前是 `decoder_batched_rowwise_visual`：视觉编码、Crop/TGVF observation 构造和 bundle rehydration 仍逐 row，非 fused 路径的 LM head/selected-logprob 也仍逐 row，尚不是 full multimodal batching。训练 loss 的全局 token 归一化是严格的，但现有 clipfrac/KL 监控仍是可变 micro 的局部均值再聚合，不能冒充严格的全局 token-weighted 指标。 |
| 3 | Remove padding + gradient checkpointing | `use_remove_padding` 和 `enable_gradient_checkpointing` 从统一 performance binding 进入 actor/model 配置；decoder batching 按各行原始长度切回输出。 | batched decoder 的输入仍按 microbatch 最长序列右侧补齐，逐 row 的视觉/工具 observation 构造也仍存在；因此 custom replay 是否实际获得 remove-padding 收益必须由 GPU canary 测量。gradient checkpointing 的节省/重算比例同样不能只从配置推断。 |
| 4 | Prefix cache + chunked prefill | v2 显式控制 vLLM prefix caching 与 chunked prefill；prefix cache 只有携带 multimodal hash identity 和 weight-update invalidation receipt 时才被兼容校验器接受。Qwen3 vLLM plugin 将处理后的 `mm_info.hashes` 传入 `MultiModalInputs.mm_hashes`。 | cache key 依赖 multimodal content hash，而不是只看文本 prefix；权重更新后必须清除 KV/prefix cache，不能跨 policy version 复用。内容哈希本身可能带来 CPU 成本，且真实命中率、更新后的失效行为尚未做 GPU/Ray canary。`mm_processor_cache_gb=0` 的独立边界没有因开启 prefix cache 而放松。 |
| 5 | Async reward workers + 可配置 judge | 规则可判样本保持本地路径；需要语义 judge 时，可由每个 AgentLoop worker 拥有的 bounded dedicated thread pool 异步执行，provider failure 仍按原 fail-closed/sample-failure contract 传播。Stage3 的独立 answer/visual judge 可以并发等待。 | `judge_max_concurrency_per_worker` 是**每个 AgentLoop worker 进程**的上限，不是全局上限；若 worker 数为 `W`、每 worker cap 为 `C`，进程侧理论并发上界可到 `W × C`，外部服务仍可能更低。默认 reward 模型路线保持 Qwen2.5-72B。换成 small/alternate judge 会改变 reward semantics，必须显式选择 `explicit_alternate`，同时绑定准确 model name、artifact identity、model-bound verifier digest 和 `alternate_semantics_acknowledged=true`；它不是可静默替换的性能参数。 |
| 6 | TP=1 多 rollout replicas + CUDA Graph | `vllm_tensor_parallel_size`、CUDA Graph 开关和严格递增的 capture sizes 由 v2 配置进入 vLLM；当前拓扑还显式绑定 rollout DP=1、PP=1、disaggregation=false，并在 receipt 中记录完整 footprint 与派生 replica 数。 | pinned veRL 的一般公式是 `world_size / (TP × DP × PP)`，disaggregation 开启时还需计入 prefill/decode footprint；当前 v2 对后者 fail closed。因此 TP=1、DP=PP=1、world size=8 表示 8 个 rollout replicas，而不是一份 TP=8 engine。未来若实验 DP/PP 或 disaggregation，应扩展 topology schema，而不能只改 TP 或沿用该 receipt。CUDA Graph 的 capture、显存占用、shape coverage 和真实收益仍需 GPU 验证。 |
| 7 | Rollout-logprob bypass | `rollout_logprob_bypass=true` 进入 actor 和 algorithm 的 rollout-correction bypass。由于 pinned veRL 会把 loss registry 名改为 `bypass_mode`，本项目在该实际 registry 名下按 fixed/dynamic 配置分派到对应的严格 loss identity。 | bypass 只跳过 rollout 后对 old-policy logprob 的再次计算；rollout 仍收集并保留采样变换后的 actual behavior logprobs，exact replay 继续用这些行为证据。它不是“关闭 logprob”。当前 v2 schema 只接受 `true`，不是一个可直接关闭的 ablation：veRL 的 non-bypass old-logprob 路径会强制请求 entropy，而当前 exact-replay engine 明确不产 entropy。若要支持 `false`，必须先实现并验证该能力，再放宽合同。这也是本项目特有的 exact-replay 优化；VTool-R1 公开 7B recipe 的 bypass 默认值为 `false`。 |

## 统一 v2 performance 接口

下面只是可嵌入完整 run config 的接口片段，不是一份已授权的训练配置：

```toml
schema_version = "policy-e2e-method-matrix-run-config-v2"

[performance]
dynamic_token_batching = true
use_remove_padding = true
enable_gradient_checkpointing = true
vllm_enable_prefix_caching = true
vllm_enable_chunked_prefill = true
vllm_enable_cuda_graph = true
vllm_cuda_graph_capture_sizes = [1, 2, 4, 8, 16]
vllm_tensor_parallel_size = 1
rollout_logprob_bypass = true
reference_replay_mode = "off"
judge_dispatch_mode = "dedicated_thread_pool"
judge_max_concurrency_per_worker = 8
```

其中 `rollout_logprob_bypass` 虽然被显式记录以防 launcher 漂移，当前仍是 required-true capability contract；它尚不是可在同一 schema 中自由切换的普通性能开关。`reference_replay_mode = "off"` 表示 KL 为零时不再为全量诊断额外执行 frozen-reference replay；需要审计时可显式改为 `full_diagnostic`，但必须作为运行性能身份的一部分记录。

本阶段后续 50-step 正式 run 已选定下面的生成与 checkpoint 参数：

```toml
[sampling]
max_response_length = 8192

[training]
maximum_optimizer_steps = 50
checkpoint_steps = [0, 10, 20, 30, 40, 50]
```

`8192` 是完整 multi-turn trajectory 的累计 policy-token 上限；它不等于 veRL 的 response transport width。后者仍由 `capacity.vllm_max_model_len - capacity.max_prompt_length` 派生，并且必须严格大于 `8192`，为环境拥有的 tool observation token 留出空间。checkpoint cadence 同样由 run config 显式拥有，不在通用 runtime 中写死；旧 run 的历史 TOML 与结果身份不回写。

## KL=0、reference replay 与 checkpoint/replay 新路径

本项目把数学上的 KL 项和可选的 frozen-reference 诊断分开管理。目标 v2 配方在 KL 系数为零且 `reference_replay_mode = "off"` 时，不再构造 reference engine，也不执行 reference exact replay；当前策略仍执行一次可求导的 exact replay。若以后开启非零 KL，则必须恢复独立的 frozen-reference no-grad forward：它不能和 current-policy forward 合成同一次模型计算，但 current 和 reference 各自都可以在自己的 microbatch 内使用 decoder batching。

这里需要区分 VTool-R1 的两个公开版本：原始论文配方使用非零 KL，而当前 `training-v2` 公开 7B recipe 的默认 actor/algorithm 配置不要求 reference policy。因而“VTool 开 KL”和“当前 recipe 可省 reference”并不冲突，它们对应不同版本。本文档的目标配置选择后者；需要 KL ablation 时必须显式建立另一份运行身份，不能只在进程中临时改系数。

checkpoint 仍由 `[training].checkpoint_steps` 选择，launcher 将离散 schedule 的最大公约数映射为上游 `save_freq`，项目 hook 再精确过滤目标 step。checkpoint step 的顺序现在是：一次正常 actor→rollout 权重同步，vLLM level-1 retained-weight sleep，提交 checkpoint，然后 wake；不再为了 checkpoint 做第二次完整权重同步。旧 manager 若声明必须二次同步会在第一次同步前 fail closed。该路径的 CPU 生命周期测试已覆盖成功、checkpoint 失败后 wake 和旧 manager 拒绝；level-1 host-RAM 峰值与真实 vLLM 恢复仍待 GPU canary。

exact replay 的本轮 batching 边界是：每行 bundle、token ownership、prompt/response、视觉位置和 observation 仍单独校验与重建；只有 injected decoder inputs 合批，输出再按原长度和 policy-owned token 位置拆回。2D attention/no-cache 已覆盖，4D mask 与 cache replay 会 fail closed。`use_fused_kernels` 目前继续保持 `false`：selected-logprob fusion 虽已有源码接线和 CPU/DTensor 局部测试，但尚未通过真实 GPU FSDP2 optimizer-step/reshard canary，也还不是 v2 独立的 config-owned 开关，因此本轮不把它冒充已激活优化。

这段配置不包含实验语义的其余必要字段。完整 run 仍必须显式绑定 prompt、parser、tool schema、observation renderer、action boundary、visual execution path、`model.image_max_pixels`、generation/token capacity、RNG、reward/scorer、dataset manifest、优化步数和输出身份。v2 把 TP、chunked-prefill 和 CUDA eager/graph 的单一真值迁到 `[performance]`；不要在旧 `[distributed]` 或 `[capacity]` 位置复制第二份相同开关。

分辨率继续由独立接口选择，例如：

```toml
[model]
image_max_pixels = 4194304 # 示例：2048^2；不是 runtime 默认值或硬编码要求
```

launcher 会把该值传到 `data.mm_processor_kwargs.max_pixels`，并将其纳入 run identity。@512 应写为 `262144`，1M 应写为约定中的明确整数；只有整组 treatment 使用同一个已验证配置和 identity，才构成统一分辨率。改变分辨率不需要改源码，但会建立新的实验合同。

## 32 prompts × 8 的准确含义

VTool-R1 公开 7B recipe 的默认值是 `TRAIN_BATCH_SIZE=32` 和 `ROLLOUT_N=8`，所以每个 rollout batch 产生：

`32 prompts × 8 trajectories/prompt = 256 trajectories`

这不等于单个 vLLM forward 的 batch size，也不等于每张 GPU 同时驻留 256 条序列。实际并发与分片还取决于 world size、TP、rollout replicas、AgentLoop worker 数、`max_num_seqs`、`max_num_batched_tokens`、token 长度和显存。对本项目而言，计划使用 `n=8` 时也必须同时检查 global prompt batch 是否确为 32，不能只看到某一层的 micro batch 或 dataset batch 就声称“32×8 已对齐”。

## Judge 路线不是纯性能开关

正常的同语义提速只修改 `[performance]` 的 dispatch mode/concurrency，并保持：

```toml
[reward]
judge_model_route = "qwen2.5_72b"
```

如果以后为了速度采用 smaller judge，则至少必须在 `[reward]` 明确写入 `judge_model_route = "explicit_alternate"`、`alternate_judge_model_name`、`alternate_judge_model_identity_sha256` 和 `alternate_semantics_acknowledged = true`，并同步使用由完整 alternate `ArtifactIdentity` 推导的 `answer_verifier` identity/digest。只换 endpoint、模型别名或性能段不会被接受。该 run 应登记为 reward-semantics intervention，不能与默认 72B judge run 组成“仅 wall-clock 不同”的效果比较。

## 与 VTool-R1 公开 recipe 的相同点和不同点

参考快照是 [training-v2 commit `d2aa283`](https://github.com/VTool-R1/training-v2/tree/d2aa28353ec10c7f91b39f502925003a81d6982d)，其中 [Qwen2.5-VL-7B chart recipe](https://github.com/VTool-R1/training-v2/blob/d2aa28353ec10c7f91b39f502925003a81d6982d/recipe/vtool/run_qwen2_5_vl_7b_chart.sh) 公开设置了 dynamic bsz、remove padding、gradient checkpointing、prefix cache、chunked prefill、TP=1、非 eager/CUDA Graph capture sizes，并保留 rollout logprob 计算。其默认 `ROLLOUT_CORRECTION_BYPASS_MODE=false`，而本项目 exact-replay 路线使用显式 bypass 并保留 actual behavior logprobs；两者不能写成同一个运行语义。

该公开脚本没有显式设置 `max_pixels`，所以仅凭脚本无法证明它与本项目的 @512、1M 或 `2048²` 图像尺度相同。若后续要声称 resolution-aligned，必须再取得并绑定 VTool-R1 实际 processor/dataset 图像预处理证据；当前只可以说性能开关方向参考了其公开 recipe。

吞吐比较还必须写清 forward 组成：VTool-R1 公开 7B recipe 默认 `rollout_correction_bypass_mode=false`，因此保留 old-policy logprob 计算，但默认不为 KL 构造 reference；本项目目标 profile 使用 `bypass=true` 且在 KL=0 时令 `reference_replay_mode=off`，因此既无 old-policy 重算，也无诊断 reference，只保留 current-policy exact replay。后者已经 decoder-batched 并复用 rollout 记录的视觉特征，但仍有逐 row observation/LM-head 开销。两者不能只凭“每步 forward 数”声称完全等价。

## CPU 验证快照

2026-09-01 在明确屏蔽 CUDA、移除外部 `OPENROUTER_API_KEY` 环境污染后完成：

- reference-off、checkpoint single-sync、decoder batching 和配置映射的联合回归：`222 passed`；
- 默认 CPU 全仓测试：`2448 passed, 3 skipped`；
- Ruff 全仓 lint 与全部 Python 变更文件的 format check：通过；
- `compileall` 与 `git diff --check`：通过。

全仓回归同时暴露并修复了一个与吞吐无关的既有兼容回归：Teacher T1 合入后的 loader 一度只接受短写 v1 verifier schema，无法读取不可变历史 JSON 使用的 namespaced v1 identity。修复仅恢复两个等价 v1 拼写的读取兼容，未改写历史 artifact 字节或 identity。三个 skip 分别来自本地 pinned veRL source 被标记为 dirty，以及缺少可选 `transfer_queue` 依赖；未使用 GPU。

## GPU 验收门槛

源码晋升与真实速度结论分开。之后获得用户明确授权并准备完整 v2 run config 时，至少应单独记录：

1. Ray/vLLM 全进程启动、TP/replica topology 和 CUDA Graph capture receipt；
2. prefix-cache 命中率以及每次 weight update 后的失效证据；
3. rollout、tool、reward、exact replay、actor update、weight sync 各阶段 wall time；
4. prompts/s、trajectories/s、policy tokens/s、GPU utilization、峰值显存和 judge queue/concurrency；
5. 32×8 的完整 batch 数、失败/重试/取消数，以及 checkpoint/resume canary；
6. fixed equal-micro control 与 dynamic global-token-mean intervention 分开报告，不将 loss reduction 差异误记为纯速度差异。
7. `reference_replay_mode=off` 时 reference worker 数与 reference forward 调用数都必须为零；
8. 记录 `replay_execution`、`decoder_batch_size`、decoder padding ratio，以及 visual/decoder/LM-head 分项耗时；
9. 普通 step 与 checkpoint step 的完整 `update_weights` 调用数都必须恰为一次，并验证 level-1 sleep/wake 前后 behavior identity 不变。

在这些证据出现前，前七项及三项热路径补强的共同结论是“接口和源码路径已对齐，CPU 合约可审计”，不是“训练吞吐已经与 VTool-R1 等价”。
