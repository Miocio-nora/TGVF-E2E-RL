# TGVF End-to-End Policy RL Pilot v1：设定、结果与分析

状态：**已完成**

实验日期：2026-07-22 至 2026-07-23

基础模型：Qwen3-VL-8B-Thinking

训练方法：80-step GRPO，language-decoder LoRA

比较方法：TGVF-only、Crop-only，以及未进行 RL 的原始 Qwen3-VL baseline

## 1. 结论摘要

这次 Pilot 的工程目标已经实现：两条 80-step GRPO 训练均完成，原生多轮工具轨迹、真实 behavior logprob、FSDP2 actor、vLLM rollout、LoRA 权重同步、checkpoint/resume、72B answer judge 和正式 VLMEvalKit 评测均实际运行。

方法结果则暂时是否定的：**当前 TGVF-only 和 Crop-only policy 都没有在 CoreDev-2511 上取得普遍优于原始 Qwen3-VL 的结果。**

- TGVF 在正式评测中频繁使用工具：73.17% 的 trajectory 至少获得一次 TGVF observation，但七项主指标均未超过 baseline。
- Crop 的工具使用率较低，为 32.86%。它在 VStar、HRBench、BLINK 和英文 OCR 上优于 TGVF，其中 HRBench 达到 58.00%，超过 baseline 2.00 个百分点。
- Crop 在 MMMU-Pro、MathVista 和 MathVerse 上明显退化，因此不能视为整体更优的方法。
- 训练期间 TGVF 的累计派生平均 reward 为 1.138，而 Crop 为 0.726；但更高的 TGVF 训练 reward 没有转化为更高的 held-out benchmark 分数。这说明当前 reward 与外部分布上的最终能力之间存在明显错位。
- 当前 conditional tool reward 使“正确且成功调用工具”的合法 trajectory 得到 2.0 reward，而“正确但直接回答”只得到 0.8 reward。这个 2.5 倍差距很可能是 TGVF 高调用率的重要驱动力，但本次 Pilot 尚不能单独证明因果关系。

因此，这次 Pilot 最合理的定位是：**端到端训练框架验证成功，第一版 reward/tool-use formulation 尚未验证出研究收益。**

## 2. 实验身份

### 2.1 两条训练 arm

| Arm | Run ID | 工具 | Step-80 LoRA SHA-256 |
|---|---|---|---|
| TGVF-only | `PRL-02-R5-QWEN3-GRPO-BS16-TGVF-T1-FORMAL-PILOT-80STEP-GPU0123` | `tgvf_focus_tool` | `561132e49848fd43f8e7f352ef54782249aff59b2a5d331027a0e5e0f78be321` |
| Crop-only | `PRL-03-R2-QWEN3-GRPO-BS16-CROP-ONLY-FORMAL-COMPARISON-80STEP-GPU0123` | `image_zoom_in_tool` | `eed4ffeaf5b77277a41dafeba428a20d5f3c8bce73049c02e63f63292d78b0b0` |

两条 arm 均从原始 Qwen3-VL-8B-Thinking reasoning policy 初始化，不使用额外 policy SFT，也不从历史 policy checkpoint 初始化。

### 2.2 TGVF representation artifact

TGVF-only arm 使用冻结的 representation phase artifact：

- artifact：`REP-QWEN3-V4-CONTEXTUAL-V4`；
- 训练长度：2,000 optimizer steps；
- target-conditioning provider：`contextual_hidden_state`，最后一层；
- representation objective：Balanced Matrix CE，temperature `1.0`；
- 输出：main `D` 加三条 D-DeepStack branch；
- native DeepStack：开启；
- artifact 文件 SHA-256：`50179c709c5788d83ffc58d13dcde9e15ed448b2cf3233a5db67cb7501106e75`。

TGVF Adapter 在 policy RL phase 中保持冻结。Crop-only arm 不使用 TGVF Adapter；每次成功 crop 后，由同一 rollout replica 的冻结 Qwen vision tower 重新编码 crop，并保留 Qwen 原生 DeepStack 路径。

## 3. 共同训练设定

| 项目 | 设定 |
|---|---|
| 基础模型 | `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Thinking` |
| 图像上限 | `max_pixels = 512 × 512 = 262144` |
| 数据 | `ChenShawn/DeepEyes-Datasets-47k` |
| 数据 snapshot | `5546681e28fa2eda9f60a9ea9dd0cf291216ded3` |
| 样本数 | 47,052 |
| shuffle / rollout seed | 42 |
| global prompt batch | 16 |
| 每 prompt trajectories | 8 |
| 每 optimizer step trajectories | 128 |
| optimizer steps | 80 |
| 总 prompts | 1,280 |
| 总 trajectories | 10,240 |
| sampling | temperature 1.0、top-p 1.0、top-k disabled、无 penalty |
| trajectory response budget | 8,192 policy-generated tokens |
| 最大工具调用次数 | 4；第 5 次返回确定性的 tool error |
| optimizer | AdamW，LR `1e-5`，weight decay `0.01` |
| scheduler | cosine，warmup 0，minimum LR ratio 0.1 |
| precision | BF16 parameter/autocast，FP32 reduce/optimizer state，TF32 enabled |
| checkpoint | 计划 step 0/10/20/45/80；最终失败时额外保存可恢复 checkpoint |
| logging | console + W&B project `tgvf-policy-rl` |

### 3.1 LoRA 和冻结范围

- LoRA rank `64`、alpha `64`、dropout `0`；
- 仅作用于 36 层 language decoder 的 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj` 和 `down_proj`，共 252 个 target modules；
- 冻结 vision encoder、visual merger、原生 DeepStack 模块、TGVF Adapter、input embeddings 和 `lm_head`；
- 不进行 full fine-tuning。

### 3.2 分布式与 rollout

- 4 张 B200，物理 GPU 0–3；
- actor 使用 FSDP2，`reshard_after_forward = false`；
- actor 和 rollout colocated，训练与 rollout 交替占用同一组 GPU；
- rollout backend 为 vLLM `0.12.0`，TP=1；
- 每个 optimizer step 同步一次 LoRA 权重；
- behavior policy 是生成该 rollout batch 时的 policy snapshot，staleness 为 0；
- 保存 sampling transform 后的真实 behavior logprob；
- policy/reference replay 使用 rollout 物化的同一份视觉 observation，不由更新后的模型重算。

## 4. GRPO 数学合同

每个 prompt 的 8 条 trajectory 保持原组，不做 group filtering。组内 advantage 使用 sample standard deviation：

```text
A_i = (r_i - mean(r)) / (sample_std(r) + 1e-6)
```

如果组内 reward 完全相同，则整组 advantage 为 0。trajectory-level advantage 只广播到 policy-generated assistant tokens；模板、图像、tool response 和 padding token 不参与 policy loss。

行为比率和 clipping 为：

```text
rho_t = exp(log pi_current - log pi_behavior)
s_t = min(rho_t * A, clip(rho_t, 0.8, 1.2) * A)
```

当 `A < 0` 时使用 dual clip `max(s_t, 3A)`。最终 loss 对全部有效 policy token 做 global token mean。

其他固定项：

- policy update epochs：1；
- clip low/high：0.2/0.2；
- dual clip：3.0；
- KL reward/loss coefficient：0；
- entropy coefficient：0；
- max gradient norm：1.0；
- 不进行 over-sampling、rejection sampling 或低 reward trajectory 丢弃。

## 5. Reward 与 answer verifier

合法 trajectory 的 reward 为：

```text
reward = 0.8 × answer_reward
       + 0.2 × format_reward
       + 1.2 × conditional_tool_reward
```

- `answer_reward`：正确为 1，否则为 0；
- `format_reward`：合法为 0，非法协议或缺少合法最终答案为 -1；
- `conditional_tool_reward`：最终答案正确且至少成功获得一次当前 arm 的视觉 observation 时为 1，否则为 0；每条 trajectory 最多发放一次；
- tool error 分类记录，但不设置不同惩罚权重；非法 trajectory 仍留在原 GRPO group 中。

选择题使用规则解析/exact match。数学题和开放式 VQA 先走规则验证，无法确定时使用 Qwen2.5-72B-Instruct semantic fallback。训练 judge 通过 OpenRouter/DeepInfra 调用，temperature 0，失败时重试而不对未完成 batch 打分；它不是 RL reference policy，也不是 SDPO teacher。

当前 reward 的一个重要性质是：

| 合法结果 | Reward |
|---|---:|
| 正确、直接回答 | 0.8 |
| 正确、至少一次成功工具 observation | 2.0 |
| 错误、格式合法 | 0.0 |
| 错误、格式非法 | -0.2 |

因此工具 bonus 不是一个轻微辅助项，而是当前总 reward 的主要组成部分之一。

## 6. 训练结果

以下为 80 steps、10,240 条 on-policy trajectory 的累计指标。训练 answer reward 是训练分布上的 on-policy verifier 结果，不是 held-out benchmark accuracy。

| 指标 | TGVF-only | Crop-only |
|---|---:|---:|
| 累计 mean answer reward | 0.6084 | **0.6235** |
| 累计 mean conditional tool reward | **0.5571** | 0.1962 |
| 累计 format error rate | 8.54% | **4.36%** |
| 派生 mean total reward | **1.1382** | 0.7255 |
| 至少尝试工具的 trajectory 比率 | **94.19%** | 51.93% |
| 平均工具调用尝试数 | **1.241** | 0.713 |
| 工具调用尝试总数 | 12,707 | 7,305 |
| 成功 observation 总数 | **10,893** | 4,167 |
| observation / attempt | **85.72%** | 57.04% |
| 平均 reasoning token 数 | 1,341.0 | 1,338.2 |
| 生成 policy token 总数 | 16.94M | 15.74M |
| 平均原图 visual token | 198.9 | 198.9 |
| 平均 trajectory visual token | 409.6 | 239.2 |
| 72B judge calls | 6,919 | 6,928 |
| 72B judge API 记录成本 | $0.6210 | $0.6385 |
| 完成 step 的 active wall-time 总和 | 4.35 h | 4.57 h |
| median step time | 174.7 s | 179.8 s |

Active wall-time 是 80 条已完成 step metric 的总和，不包含人工 debug、进程重启或训练暂停的日历时间。

### 6.1 训练行为变化

| Arm | 指标 | Step 1 | Step 80 |
|---|---|---:|---:|
| TGVF | tool-attempt trajectory rate | 89.06% | 100.00% |
| TGVF | mean reasoning tokens | 1,790.6 | 963.8 |
| Crop | tool-attempt trajectory rate | 39.84% | 69.53% |
| Crop | mean reasoning tokens | 1,487.8 | 1,168.2 |

这些是两个不同 16-prompt batch 的观测，不是平滑 learning curve，因此不应把单个 step 的变化解释为统计显著趋势。不过结合累计指标，可以确认两条 policy 都向更高工具使用率移动，且 TGVF 的移动更强。reasoning 变短也不能直接解释为“模型更确定”；后续 benchmark 退化说明它也可能包含压缩、过早作答或 reward-induced behavior change。

### 6.2 训练期间的主要错误

- TGVF 累计出现 1,369 次 `missing_think_closer`、219 次调用上限错误、132 次 invalid JSON，以及少量其他 protocol parse error。
- Crop 累计出现 2,500 次 `tool_execution_failed`，另有 403 次 `missing_think_closer`、113 次 incomplete tool call 和 84 次调用上限错误。
- Crop 的 57.04% observation/attempt 比率说明 crop 执行稳定性是这条 arm 的实质性问题，而不仅是日志噪声。
- 两条训练都实际验证了 checkpoint/resume。TGVF 在上游 judge 持续失败时额外保存过 step-76 恢复点，之后完成到 step 80。

## 7. 正式评测设定

### 7.1 CoreDev-2511

正式评测使用固定的 CoreDev-2511/VLMEvalKit snapshot：

- VStarBench：191；
- HRBench4K：200；
- BLINK：420，其中当前工具 runtime 支持 180 条单图题；
- OCRBench_v2：600；
- MMMU-Pro 10-choice：300，其中当前工具 runtime 支持 269 条单图题；
- MathVista MINI：300；
- MathVerse MINI：500。

TGVF/Crop 共完成 2,240 条单图 trajectory。BLINK 的 240 条和 MMMU 的 31 条多图题没有静默取第一张图或拼图，而是作为 unsupported fail-closed。主结果表对 BLINK/MMMU 使用 baseline、TGVF、Crop 共同覆盖的单图行，避免把接口未支持直接算成模型错误。

所有 arm 使用同一固定数据 membership 和 `max_pixels = 262144`。正式评分基于 pinned VLMEvalKit commit `7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f`；需要 judge 的 benchmark 使用本地 Qwen2.5-72B-Instruct，未使用 GPT。

### 7.2 Baseline 比较口径限制

原始 Qwen3-VL baseline 与两个 RL arm 使用同一个基础模型、图像上限和正式 scorer，但不是严格的 decoding-controlled ablation：

- baseline：temperature 1.0、top-p 0.95、top-k 20、max new tokens 40,960；
- RL arms：temperature 1.0、top-p 1.0、top-k disabled、完整多轮 trajectory budget 8,192；
- baseline 使用直接 benchmark prompt；RL arms 使用各自的 native visual-tool system prompt。

因此，baseline 表反映的是三套系统的端到端 utility 对比，不能把全部差值严格归因于 LoRA 或工具本身。TGVF 与 Crop 两条 arm 之间的 sampling/data/hardware 设定则基本一致，是更直接的 arm-to-arm 比较。

## 8. CoreDev benchmark 结果

| Benchmark | 原始 Qwen3 baseline | TGVF-only | Crop-only | 最佳 Pilot arm 相对 baseline |
|---|---:|---:|---:|---:|
| VStarBench | **53.93%** | 51.31% | 53.40% | -0.52 pp |
| HRBench4K | 56.00% | 54.00% | **58.00%** | **+2.00 pp** |
| BLINK 单图 180 题 | **64.44%** | 60.00% | 61.11% | -3.33 pp |
| OCRBench English | **49.78%** | 41.61% | 47.53% | -2.25 pp |
| OCRBench Chinese | **39.38%** | 34.44% | 34.25% | -4.94 pp |
| MMMU-Pro 单图 269 题 | **63.94%** | 48.33% | 43.87% | -15.61 pp |
| MathVista MINI | **77.33%** | 69.67% | 68.67% | -7.67 pp |
| MathVerse Text Dominant | **75.00%** | 66.00% | 53.00% | -9.00 pp |

除 HRBench 外，原始 Qwen3 baseline 在所有报告指标上均为最佳。TGVF 没有在任一项主指标超过 baseline；Crop 仅在 HRBench 上超过 baseline。

### 8.1 正式评测中的工具行为

| 指标（2,240 条单图） | TGVF-only | Crop-only |
|---|---:|---:|
| 至少一次成功 observation | **73.17%** | 32.86% |
| 平均成功 observation/trajectory | **0.949** | 0.411 |
| final answer after tool path | 1,658 | 842 |
| direct answer | 377 | 1,130 |
| max-tokens stop | 148 | 163 |
| call-cap stop | 48 | 99 |
| invalid-format stop | 9 | 6 |

评测工具率低于训练累计工具率，说明 DeepEyes 训练分布与 CoreDev benchmark 的问题类型、prompt 或 policy routing 行为存在分布差异。

## 9. 分析

### 9.1 学会调用工具不等于学会利用工具

TGVF 的正式评测工具率达到 73.17%，说明“policy 不会调用 TGVF”不是当前主要问题。但它没有带来 benchmark 增益，说明至少还存在以下一种或多种情况：

1. target 生成不够具体，或者包含未经图像验证的猜测；
2. TGVF observation 包含信息，但 policy 没有稳定读取并用于更新答案；
3. observation 在 DeepEyes 训练分布上有用，在 benchmark 分布上泛化不足；
4. 工具调用打断或压缩了原始 Qwen reasoning；
5. reward 主要优化了“正确时调用过工具”，而不是“工具 observation 对答案产生了可验证的因果贡献”。

本次结果不能区分以上机制，需要 trajectory-level controlled analysis 和更直接的 counterfactual evaluation。

### 9.2 当前 conditional tool bonus 可能过强

TGVF 的累计 answer reward 略低于 Crop，但派生 total reward 高出约 0.413，主要来自 conditional tool reward。与此同时，TGVF benchmark 并未更好。

这说明当前 reward 很容易把“调用工具”本身与“有价值地调用工具”混在一起。下一版不应简单取消所有工具激励，但需要验证以下更精确的信号：

- target 是否是自包含、可视觉验证且不泄漏猜测答案；
- observation 是否改变了答案或提高了正确答案概率；
- 工具调用相对 direct path 是否产生正的 counterfactual gain；
- 多次调用是否提供互补证据，而不是重复或无关调用。

在这些信号被接受前，不应直接增加新的 target reward 权重，因为一个未经校准的 judge reward 也可能产生新的 reward hacking。

### 9.3 Crop 的局部优势与推理退化同时存在

Crop 相对 TGVF 在 VStar、HRBench、BLINK 和英文 OCR 上分别提高 2.09、4.00、1.11 和 5.92 个百分点，符合 crop 对小目标、文字和局部属性任务可能更直接的预期。

但 Crop 在 MMMU-Pro、MathVista 和 MathVerse 上分别比 TGVF 低 4.46、1.00 和 13.00 个百分点。它不是稳定更优的替代方案。可能原因包括：

- crop policy 更少调用工具，因此局部任务接近原始 policy，而复杂任务上的 LoRA 改变仍损害 reasoning；
- crop 会丢失全局上下文，且 relation/comparison 题对 bbox 选择更敏感；
- 训练期间大量 crop execution failure 降低了有效 tool-positive trajectory 数；
- 当前 reward 没有区分“需要全图推理”与“适合局部放大”的样本。

### 9.4 Reasoning 变短不能解释为更确定

TGVF 和 Crop 的 step-80 batch reasoning 都比 step-1 batch 更短。若只看训练轨迹，这可能被解释为效率提高；但 benchmark 普遍下降，尤其是 MathVerse/MMMU。因此当前证据更支持保守结论：**reasoning shortening 是一个需要监控的行为变化，不是能力提升证据。**

### 9.5 训练 reward 与 held-out utility 不一致

Crop 的累计 answer reward 为 0.6235，TGVF 为 0.6084；但两者在多数 benchmark 上均低于原始 policy。训练 reward 来自 DeepEyes on-policy 样本，并且混合了规则解析、72B semantic fallback 和工具 bonus。它不能替代独立 benchmark，也不能直接代表 general reasoning retention。

## 10. 当前限制

1. 仅正式评测了 step 80；虽然保存了 step 0/10/20/45/80 checkpoint，但还没有形成完整的 benchmark-vs-step curve。
2. 原始 baseline 与 RL arms 的 decoding budget/prompt 不完全一致，尚缺同 sampling、同 response budget 的 step-0 controlled baseline。
3. TGVF/Crop 工具接口尚不支持多图样本的 `image_index`，因此正式可比覆盖为 2,240/2,511。
4. Pilot 只训练 decoder LoRA，不代表 full fine-tuning 结论。
5. 仅训练 GRPO；没有 SDPO、GRPO+SDPO、非零 KL 或 TGVF+Crop fusion。
6. 没有对 target quality、observation causal contribution 或 reasoning retention 设置独立 reward。
7. Crop 训练的工具执行失败率较高，需要在解释方法能力前先拆分 bbox policy error、runtime error 和图像处理 error。
8. 评测结果来自固定小型 CoreDev slice，适合快速方法筛选，不替代完整 benchmark evaluation。

## 11. 建议的下一步

按优先级建议：

1. **补受控 step-0 baseline**：相同 sampling、8,192 budget、相同 agent runtime，分别测 direct、TGVF 和 Crop，以拆分 prompt/runtime 与 RL 权重影响。
2. **做 checkpoint curve**：至少评测 step 0/20/45/80 的代表性小切片，确认退化从何时开始，避免只比较最终点。
3. **审计工具因果贡献**：对同一 trajectory 保存 direct、真实 observation、错配 observation 和屏蔽 observation 的可比 replay，测 answer/logprob 变化。
4. **审计 target quality**：建立小规模人工审计集，分类 target 是否具体、是否含猜测答案、是否需要局部/全局证据，以及 observation 是否回答了 target。
5. **修复并分类 Crop failures**：在扩大训练前先降低 `tool_execution_failed`，分别统计无效 bbox、越界、空 crop、processor/runtime failure。
6. **重新校准 reward**：在上述审计数据上决定 conditional tool bonus 是否降权，以及是否加入经过验证的 target/causal observation reward。
7. **最后再做 TGVF+Crop fusion**：只有单工具路由和 reward 能被解释后，再训练融合 arm，否则难以判断收益来源。

## 12. 复现实物与结果路径

- TGVF 训练配置：[`../configs/policy/runs/prl_02_r5_qwen3_grpo_bs16_tgvf_t1_formal_pilot_80step_gpu0123.toml`](../configs/policy/runs/prl_02_r5_qwen3_grpo_bs16_tgvf_t1_formal_pilot_80step_gpu0123.toml)
- Crop 训练配置：[`../configs/policy/runs/prl_03_r2_qwen3_grpo_bs16_crop_only_formal_comparison_80step_gpu0123.toml`](../configs/policy/runs/prl_03_r2_qwen3_grpo_bs16_crop_only_formal_comparison_80step_gpu0123.toml)
- TGVF 累计训练指标：`artifacts/policy/PRL-02-R5-qwen3-grpo-bs16-tgvf-t1-formal-pilot-80step-gpu0123/metrics.jsonl`
- Crop 累计训练指标：`artifacts/policy/PRL-03-R2-qwen3-grpo-bs16-crop-only-formal-comparison-80step-gpu0123/metrics.jsonl`
- 原始 Qwen baseline 汇总：`artifacts/evaluation/BE-03-qwen3-direct-coredev2511-final-answer-j1/coredev-2511-eval-summary.json`
- TGVF 正式评测汇总：`artifacts/evaluation/BE-04-qwen3-tgvf-step80-coredev2511-gpu0123/scoring/tgvf/coredev-2511-eval-summary.json`
- Crop 正式评测汇总：`artifacts/evaluation/BE-05-qwen3-crop-step80-coredev2511-gpu0123/scoring/crop/coredev-2511-eval-summary.json`
