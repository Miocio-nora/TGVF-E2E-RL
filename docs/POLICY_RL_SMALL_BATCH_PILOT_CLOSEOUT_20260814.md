# Policy RL 小批量 Pilot 收官：Crop、TGVF 与组合工具

日期：2026-08-14

状态：`PILOT PHASE CLOSED / SUCCESS`

Decision ID：`POLICY-RL-SMALL-BATCH-PILOT-CLOSEOUT-20260814-v1`

范围：`Qwen3-VL-8B-Instruct`、T1 mixed-v2、BS16 × n16、PRL13--PRL20。

## 1. 收官结论

本轮 small-batch pilot 到此正式结束，并且是一次成功的 pilot。成功并不意味着每个
实验 arm 都单调提升，而是我们已经用可运行、可恢复、可外部评测的完整闭环回答了
进入下一阶段前最关键的问题：

1. **Crop policy-RL 可以学习。** Crop clean-final 从 Step 0 的 Macro* `55.5742`
   提升到 Step 8 的 `59.7161`，Step 16 为 `59.5502`；当前规模下 Step 8--16
   更像平台，而不是持续退化。
2. **TGVF 的 `D` 确实有答题效用。** 独立 Stage1 paired utility 实验中，
   image-only 为 `72.5%`，image + correct-D 为 `83.5%`；相对 zero-D / wrong-D
   的受控增益仍为 `+7.5 / +6.5 pp`。`D` 的主要角色是原图上的
   target-focused complementary residual，不是原图替代品。
3. **TGVF policy-RL 也出现正向信号。** RP67 T-free Frozen 的 paired Macro*
   为 `57.0320 → 56.1964 → 58.1996`（Step 0/8/16），Step 16 相对初始化
   `+1.1675 pp`。单 seed 下不能称为统计确认，但足以说明该线路并非无效。
4. **当前 BS16 下应冻结 TGVF Adapter。** 当前 single-seed、matched-training、
   paired-evaluation 对照中，Frozen Step 16 比 Joint/Unfrozen Step 16 高
   `1.6713 pp`；Joint Step 16 还低于共同 Step 0。这强力支持当前默认冻结，
   但不是统计确认，也不外推到更大 batch。
5. **当前工程默认是 RP67 + T-free + Frozen。** RP67 是目前最受支持的 Stage1
   主线；移除表示依赖的 counterfactual utility `T` 后，reward 更简单、控制更干净；
   Adapter 冻结，只更新 full Qwen policy，包括 vision encoder、merger 与 LM。
6. **Crop 与 TGVF 可以兼容。** Atomic Crop+TGVF 在 Step 8 达到本轮观察到的最高
   CoreDev Macro* `62.1168`，Step 16 为 `60.9539`。这证明组合工具可以稳定工作并
   保持较强外部能力；由于不同线路没有共享同一 prompt/tool/RNG block，不能把跨线路
   差值写成严格的 synergy 因果效应。
7. **Target/Focus 与 Grounding reward 值得保留，但暂不进入默认配方。** 手工抽查
   观察到 reasoning 中的看图描述更具体、更贴图；accuracy 在 Step 8 有正向差异，
   Step 16 又落后于 no-visual control。该性质尚缺独立 held-out foveation/
   hallucination audit，因此不能写成“已证明有效”或“已证明无效”。
8. **BS16 的更新噪声是当前最有力的限制解释，但尚不是已完成的 batch ablation
   结论。** 当前 reward 与联合 recipe 有学习信号；非单调变化更像少量独立 prompt
   group 下的梯度方向噪声、稀疏 credit 与能力重分配，而不是简单的“RL 无效”。

因此，本阶段不再继续在 BS16 上排列组合相近的 reward/结构变体。下面冻结的默认线
可作为后续工作的可靠起点；更大 batch、视觉健康度和其他新方向作为新的、独立命名
阶段处理。

## 2. 证据边界与统一测量口径

正式 accuracy 均遵循
[CoreDev-2511 统一测量合同](POLICY_RL_COREDEV2511_MEASUREMENT_CONTRACT_AND_BASELINES_20260812.md)：

| 字段 | 固定口径 |
|---|---|
| suite | CoreDev-2511，共 2,511 rows |
| 实际工具推理 | 2,240 条单图 |
| 显式 hold | 271 条多图：BLINK 240、MMMU-Pro 31 |
| sampling | `temperature=1`、`top_p=1`、每题一次采样 |
| response / call cap | 20,480 tokens / 最多 6 次工具调用 |
| final-answer dialect | plain text；不使用 `<answer>...</answer>` |
| benchmark scorer | VLMEvalKit；语义题使用本地 Qwen2.5-72B-Instruct |
| headline | 七个等权分量的非官方诊断均值 Macro* |
| paired block | `paired-seed-v1`、master seed 42、common random numbers per task turn |

Macro* 的七个分量是 VStar、HRBench Average/all、BLINK 单图子集、OCR EN/CN
均值、MMMU-Pro 单图子集、MathVista 与 MathVerse five-version macro。

必须保留以下比较边界：

- Original 是 raw direct prompt，只能作为端到端参考，不是工具 arm 的严格 control；
- Crop S0/S8/S16 是同测量口径的 `legacy-RNG` 历史块；
- RP67 Frozen/Joint/Visual 各结果使用明确的 paired block；
- Crop+TGVF 只在自身 Step 8/16 内共享 paired RNG；它没有组合工具 Step 0；
- legacy 与 paired、不同 prompt/tool schema 或不同 seed namespace 之间，不计算无标签的
  因果 delta；
- `temperature=1` 即便 paired 仍是单 seed、每题一次采样。约 `1 pp` 的变化必须按
  “趋势/信号”表达，不能冒充统计显著性；
- 2026-08-12 的 `temperature=0` run 改变了模型失败模式并产生大量机械重复，属于
  greedy-stability stress diagnostic，不进入 accuracy 主表。

## 3. 固定的数据与训练设置

### 3.1 T1 mixed-v2 数据

候选样本由原始 `Qwen3-VL-8B-Instruct` 在 full-image、无工具条件下各生成 8 条
rollout。T1 保留天然答对 `1..7/8` 的中间难度样本，排除 `0/8` 与 `8/8`。

| source | candidates | retained | retain ratio |
|---|---:|---:|---:|
| VStar | 170,000 | 39,205 | 23.06% |
| ArxivQA | 32,000 | 25,393 | 79.35% |
| ThinkLite | 69,842 | 12,943 | 18.53% |
| **total** | **271,842** | **77,541** | **28.52%** |

权威数据根为：

```text
artifacts/data/policy_rl/
  T1-04-INSTRUCT-FULL-MIXED-T1-RETAINED-FINAL-v2/
```

```text
manifest_file_sha256 = 752ebe9ea5fced48773b9bc0babfbb6bc57a335dd1b580455f6962053d29fddf
content_sha256       = 5ab99622a2698a7c52c45795215fa5c467b741c103827a1a7dbe3800ff052934
samples_sha256       = 06e5b1b9039680111df5ef01f7f969b9cf3d8d0eaefa5774fd8d16169428611a
schedule_seed        = 42
```

### 3.2 当前 matched RL 配方

| 字段 | 固定值 |
|---|---|
| policy | `Qwen3-VL-8B-Instruct` |
| algorithm | GRPO；每组 16 trajectories |
| prompt batch | 16 independent prompt groups / optimizer update |
| total trajectories | 256 / optimizer update |
| distributed | world8、prompt micro2/rank、GA1、FSDP2 |
| trainable Qwen | vision encoder + merger + language model，全部更新 |
| representation | RP67 Step 2000；默认 frozen |
| optimizer | AdamW，LR `1e-6`，weight decay `0.01` |
| scheduler | constant，无 warmup |
| gradient clipping | `1.0` |
| PPO epochs / KL | 1 / 0 |
| sampling | temperature 1、n16 |
| horizon | Step 8 是首个 scale gate；Step 16 用于继续性判断 |

“Frozen Adapter”不等于完全冻结 observation generator：共享的 Qwen vision path
仍随 policy 更新。它表示 72,055,808 参数的 RP67 Adapter 本身不接收 policy-RL
梯度。

## 4. 工具与 reward 线路

| 线路 | policy action | 返回给 policy 的 observation | 本轮 reward |
|---|---|---|---|
| Crop | `image_zoom_in_tool(bbox_2d, label?)` | 不可变原图上的真实 RGB crop | 历史 DeepEyes-style answer + format + answer-gated tool reward |
| TGVF | `tgvf_focus_tool(target)` | 原图条件的 latent `D` 与 D-DeepStack | 默认 T-free：answer + repeated-call + protocol/tool-error penalty |
| Crop+TGVF | `tgvf_crop_tool(bbox_2d, target)` | crop 经 Qwen vision 与 frozen RP67 后的 `D`；不返回 RGB crop | 与 TGVF 相同的 T-free reward |

Crop clean-final 的视觉样本历史 reward 为：

```text
R_crop = 0.8 × AnswerCorrect
       + 0.2 × FormatScore
       + 1.2 × AnswerCorrect × HasSuccessfulCrop
```

该历史 Crop run 中 ThinkLite 是 direct-only/no-tool 分支，reward 为
`1.2 × AnswerCorrect + 0.4 × FormatScore`。

当前 TGVF / Crop+TGVF 默认 T-free reward 为：

```text
R_tfree = 2 × AnswerCorrect
        − 0.05 × max(0, ToolCallCount − 1)
        + ProtocolOrToolErrorPenalty

ProtocolOrToolErrorPenalty = −1 if an error occurred, otherwise 0
```

这里的 `T-free` 是移除旧 counterfactual tool-utility sidecar `T`，不是移除 tool
target，也不是禁用工具。`T` 的标签依赖表示版本，RP66 标签迁移到 RP67 会造成 reward
错配；重算标签又会同时改变 reward，因而没有作为当前默认继续保留。

PRL19 只在 `R_tfree` 上额外加入 Focus/Target `F` 和 Grounding `G`：

```text
R_visual = R_tfree + F + G
```

一次 gold-free OpenRouter 调用使用固定的 `qwen/qwen3-vl-32b-instruct`，输入原图、
问题、按顺序的成功 targets、tool 后 reasoning 与最终答案，不含 gold/reference answer。
`F` 的 `2/1/0` 映射为 `1/0.5/0`；`G` 映射为 `1/0.5/−1`。训练 answer verifier
仍为 rule-first + Qwen2.5-72B semantic fallback；CoreDev 正式语义 scorer 使用本地
Qwen2.5-72B-Instruct。

## 5. Canonical 结果

### 5.1 一页总表

下表中的 delta 只在同一 RNG/protocol block 内解释。`—` 表示未测，不能补成其他
线路的 Step 0。

| 线路 / block | Step 0 | Step 8 | Step 16 | 有效的 block 内结论 |
|---|---:|---:|---:|---|
| Original direct | **55.36** | — | — | 端到端 direct reference；非工具 paired control |
| Crop clean-final / legacy-RNG | 55.5742 | **59.7161** | 59.5502 | S8−S0 `+4.1419 pp`；S16−S0 `+3.9760 pp` |
| RP67 T-free Frozen / paired | 57.0320 | 56.1964 | **58.1996** | S16−S0 `+1.1675 pp`；S16−S8 `+2.0032 pp` |
| RP67 T-free Joint / paired | 57.0320 | 56.2035 | 56.5283 | S16−S0 `−0.5038 pp` |
| RP67 Frozen + F/G / paired | 57.0320 | **57.8849** | 57.5422 | S8 比 matched no-F/G S8 `+1.6885 pp`；S16 比 matched no-F/G S16 `−0.6573 pp` |
| RP67 Frozen T-free Crop+TGVF / own paired | — | **62.1168** | 60.9539 | S16−S8 `−1.1629 pp` |

### 5.2 Crop clean-final

| benchmark | Crop S0 | Crop S8 | Crop S16 |
|---|---:|---:|---:|
| VStarBench | 78.01 | 76.96 | 79.06 |
| HRBench Average/all | 53.50 | 62.50 | 60.00 |
| BLINK single-image | 57.22 | 60.00 | 57.78 |
| OCR EN/CN mean | 38.95 | 49.30 | 48.95 |
| MMMU-Pro single-image | 43.87 | 47.58 | 50.93 |
| MathVista | 62.67 | 67.67 | 65.33 |
| MathVerse five-version | 54.80 | 54.00 | 54.80 |
| **Macro\*** | **55.5742** | **59.7161** | **59.5502** |

Crop 的主要事实是同协议 Step 0→8 广泛提升，而非所有 benchmark 都支配 Original。
相对 direct Original，Crop 在 VStar 上显示出很强的 pipeline 能力；Original 与 Crop
使用不同 prompt/tool 协议，因此不能把差值全部归因于 crop pixels。

### 5.3 RP67 Frozen、Joint 与 Visual reward

| benchmark | Common S0 | Frozen S8 | Frozen S16 | Joint S8 | Joint S16 | Visual S8 | Visual S16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| VStarBench | 62.83 | 65.45 | 64.92 | 64.40 | 57.59 | 68.06 | 66.49 |
| HRBench Average/all | 58.50 | 60.00 | 64.50 | 61.50 | 67.00 | 63.50 | 60.00 |
| BLINK single-image | 62.78 | 60.56 | 63.33 | 64.44 | 62.22 | 63.33 | 60.56 |
| OCR EN/CN mean | 43.53 | 39.39 | 40.89 | 36.36 | 36.28 | 41.98 | 43.65 |
| MMMU-Pro single-image | 50.19 | 47.58 | 47.96 | 47.58 | 49.07 | 44.98 | 48.70 |
| MathVista | 68.00 | 68.00 | 70.00 | 67.33 | 68.33 | 67.33 | 69.00 |
| MathVerse five-version | 53.40 | 52.40 | 55.80 | 51.80 | 55.20 | 56.00 | 54.40 |
| **Macro\*** | **57.0320** | **56.1964** | **58.1996** | **56.2035** | **56.5283** | **57.8849** | **57.5422** |

Frozen 与 Joint 是当前最干净的 Adapter update 对照：两者 Step 8 几乎相同，
但 Frozen Step 16 比 Joint Step 16 高 `1.6713 pp`。Joint 的 HR/MathVerse 增长
同时伴随 VStar/OCR 回落，属于能力重分配，而不是总体增强。

Visual reward 的 accuracy 非单调。其训练 judge 在后 8 步没有 collapse：对成功
tool trajectories 条件化后，`F` 从 `0.7982` 到 `0.8082`，`G` 从 `0.7286` 到
`0.7717`；但 answer correctness 同时从 `0.6860` 降到 `0.5991`。由于 `F+G`
最大值可达到 2，和正确答案贡献 `2×A` 同量级，scalar 可能允许模型用 judge-visible
focus/grounding 换取正确率。Step 16 还出现更重的极端长度/重复 tail。这些都是后续
reward 设计需要处理的诊断，不是视觉 reward 已被判死刑。

### 5.4 Atomic Crop+TGVF

| benchmark | Combo S8 | Combo S16 | S16−S8 |
|---|---:|---:|---:|
| VStarBench | 70.1571 | 71.7277 | +1.5707 |
| HRBench Average/all | 73.0000 | 68.0000 | −5.0000 |
| BLINK single-image | 65.0000 | 61.6667 | −3.3333 |
| OCR EN/CN mean | 53.4388 | 52.4690 | −0.9698 |
| MMMU-Pro single-image | 47.9554 | 49.8141 | +1.8587 |
| MathVista | 69.6667 | 67.0000 | −2.6667 |
| MathVerse five-version | 55.6000 | 56.0000 | +0.4000 |
| **Macro\*** | **62.1168** | **60.9539** | **−1.1629** |

PRL20 的唯一工具 action 同时提交 bbox 与 target。runtime 从不可变原图裁剪，使用
Qwen vision + frozen RP67 生成 crop-conditioned `D`，policy 只接收 latent `D`，
不接收 crop RGB。Step 8/16 的 2,240 条单图推理均完成，271 条多图显式 hold，
sample-local evaluation failure 为 0。

作为同一测量标准下的描述性参照，Combo S8 高于 Crop S8 `2.4007 pp`、高于纯
TGVF Frozen S8 `5.9204 pp`；Combo S16 分别高 `1.4037 / 2.7543 pp`。这些不是
strict paired synergy delta，因为 Crop 是 legacy-RNG，纯 TGVF 与 Combo 的 prompt、
tool schema 和 seed namespace 也不同。合法结论是“组合兼容并显示很强的 pilot
表现”，不是“已证明二者存在精确的加性收益”。

## 6. Stage1 `D` utility：为什么可以说 TGVF 有用

RP66 的独立 first-200、十臂 paired diagnostic 给出了比 policy Step 0 横向比较更直接
的因果证据：

| context | zero-D | correct-D | wrong-D |
|---|---:|---:|---:|
| D-only fresh context | 35.0% | **65.5%** | 55.0% |
| image + D | 76.0% | **83.5%** | 77.0% |
| direct D replacement | 22.0% | **49.0%** | 46.0% |
| image-only | — | **72.5%** | — |

关键 paired 效应为：

- image + correct-D 相对 image-only：`+11.0 pp`；
- image + correct-D 相对 image + zero-D：`+7.5 pp`；
- image + correct-D 相对 image + wrong-D：`+6.5 pp`；
- direct correct-D replacement 相对 image-only：`−23.5 pp`。

因此，正确结论是：`D` 能在原图全局上下文上补充 target-specific 视觉证据；它不适合
替代完整图像。该实验使用 oracle target、first-200 和 diagnostic semantic overlay，
所以它证明的是 Stage1 representation utility，不等于 policy 已经学会在所有题上自主
选择正确 target，也不等于端到端可实现 `+11 pp`。

详细证据见
[TGVF D Answer Utility Follow-up](../reports/TGVF_D_Answer_Utility_Followup_20260730.tex)。

## 7. 证据强度与最终决策

| 判断 | 本轮状态 | 严格边界 |
|---|---|---|
| `D` 对答题有增量效用 | **已建立** | oracle-target first-200 diagnostic；证明 representation utility，不直接证明自主 tool policy |
| Crop RL 有效 | **强 positive pilot evidence** | legacy-RNG 单次评测，但 `+4.14 pp` 明显大于约 1 pp 的已观测采样波动 |
| Frozen RP67 T-free RL 有效 | **positive paired signal** | 单 seed `+1.17 pp`；支持有效，不写统计确认 |
| 当前 BS16 下 Frozen 优于 Joint | **当前 single-seed direct control 强支持** | paired evaluation 只有一个 seed；只适用于当前 batch、LR 与 horizon |
| RP67 是当前默认 | **工程决策已冻结** | 尚无 RP66/RP67 同 paired-seed 严格 representation ablation |
| T-free 是当前默认 | **最受支持的 reward 决策** | `+T`/T-free 旧比较不是共享随机流的严格单变量证明 |
| Crop+TGVF 兼容 | **强 pilot evidence** | 组合内部 S8/S16 paired；跨工具线路不是 strict synergy test |
| F/G 改善视觉 reasoning | **定性观察 + 训练信号** | 未完成独立、盲化 held-out foveation/hallucination audit |
| F/G 改善最终 accuracy | **未建立** | S8 正、S16 负；默认关闭但保留探索价值 |
| BS16 是平台期主因 | **当前最有力工作假设** | 尚无 matched larger-BS ablation，不能写成已证实因果 |

## 8. 为什么当前容易平台：batch 与 DeepEyes scale

当前每次 update 使用 `16 prompts × 16 rollouts = 256 trajectories`。`n=16` 提供了
足够的同题好坏对比，但独立问题只有 16 个；它降低的是 group 内估计噪声，不能替代
更多独立 prompt groups。

作为 scale 参照，DeepEyes 论文明确报告 `256 prompts × 16 rollouts = 4,096
trajectories/update`，共 80 iterations；论文正文没有报告 learning rate。DeepEyes
官方 released launcher 使用 `1e-6` constant LR，而我们当前也使用 `1e-6`。我们
16 个 small-batch steps 总计看过 256 prompts / 4,096 trajectories，但期间进行了
16 次参数更新；DeepEyes 是在相同数量级的 raw sample exposure 上形成一个大 batch
update。即便 nominal code LR 相同，也不能把我们的 effective LR 简单写成“16 倍”：
GRPO advantage、trajectory 长度、裁剪和梯度方向都不是线性的。

这解释了为何当前线路能快速看到方向，却容易出现 S8/S16 非单调、不同 benchmark
能力重分配和 Adapter 被 joint gradient 破坏。它是合理且与结果一致的诊断，但由于
本轮没有完成固定总 exposure 的 BS16-vs-larger-BS 对照，仍应保持为工作假设。

## 9. 本阶段冻结的默认线

```text
Base model              = Qwen3-VL-8B-Instruct
Data                    = T1 mixed-v2, 77,541 retained rows
Stage1 Adapter          = RP67 Step 2000
Adapter during RL       = frozen
Policy during RL        = full Qwen, including vision + merger + LM
Reward                  = T-free
                           answer correctness
                           + repeated-call penalty
                           + protocol/tool-error penalty
Disabled by default     = counterfactual utility T, Focus/Target, Grounding
Prompt final dialect    = plain text, no <answer> wrapper
Small-pilot checkpoints = Step 8 first gate; Step 16 continuation gate
Evaluation              = CoreDev-2511 temp1 paired-seed-v1 seven-part Macro*
```

这是一条可复用的 baseline，不是未来所有实验必须永久保持的设定。任何后续工作只要
改变 Adapter 更新、reward、batch、prompt/tool 或 sampling，就应新建独立实验身份，
并明确引用本报告作为 control。

## 10. 下一阶段边界

本轮已经完成“当前 small-batch recipe 是否可学、TGVF 是否有用、Adapter 是否应冻结、
Crop 与 TGVF 是否兼容”的验证。下一阶段可以探索其他方向，但不把尚未授权的想法写成
既定实验。当前保留的研究问题包括：

1. 固定总 prompt/trajectory exposure 的 larger-prompt-batch 对照，区分 update noise
   与真实平台；
2. 更大 batch 下 joint Adapter 是否仍会破坏 Stage1，或可安全释放联合优化能力；
3. 独立、盲化的 target quality / wrong-image sensitivity / hallucination audit；
4. 保证 answer dominance 的视觉 shaping，避免 `F+G` 抵消错误答案代价；
5. 数据侧筛选“真实使用 `D` 后有边际收益”的样本，而非只奖励调用工具；
6. 以 Crop、Frozen TGVF 和 Crop+TGVF 三条已建立 baseline 为起点，探索新的结构、
   optimization 或 credit-assignment 方向。

这些问题进入新的阶段后，应单独预注册变量和证据门槛；本 small-batch pilot 本身不再
继续扩展。

## 11. Artifact 与来源索引

### 统一测量合同

```text
docs/POLICY_RL_COREDEV2511_MEASUREMENT_CONTRACT_AND_BASELINES_20260812.md
```

### Stage1 `D` answer utility

```text
reports/TGVF_D_Answer_Utility_Followup_20260730.tex
artifacts/representation_experiments/answer_utility/
  evaluation/formal_first200_short_reader_v2_score_v3_10arm_20260730/
```

### Crop clean-final

```text
artifacts/evaluation/PRL13-A-CoreDev2511-clean-no-answer-paired-mem080-v1/
artifacts/evaluation/PRL14-A-CoreDev2511-cleanfinal-step0-step8-step16-v1/
```

### RP67 T-free Frozen

```text
artifacts/policy/
  PRL-17-R2-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-novisual-8step-ws8/
    evaluation/
      PRL17-R2-FROZEN-RP67-TFREE-COREDEV2511-STEP0-STEP8-STEP16-PAIRED-SEED-V1/
        paired-summary.json
```

`paired-summary.json` SHA256：
`bf90a99f52f1943509fa83b8c377c959d32699e5127021ea1b09c49941119176`。

### RP67 T-free Joint

```text
artifacts/policy/
  PRL-18-R0-qwen3-instruct-full-joint-rp67-bs16-n16-tfree-novisual-8step-ws8/
    evaluation/
      PRL18-R0-JOINT-RP67-TFREE-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1/
        paired-summary.json
```

`paired-summary.json` SHA256：
`2c7172f60b343f5ecf0749ade451b4617dc1c415b041a309b45b2248a81fdfaa`。

### RP67 Focus/Grounding visual reward

```text
artifacts/policy/
  PRL-19-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-visual-api-8step-ws8/
    evaluation/
      PRL19-R0-FROZEN-RP67-TFREE-VISUAL-API-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1/
        paired-summary.json
```

`paired-summary.json` SHA256：
`581c37c8e68d1dbe30e7be715e4dfd53fa8cf983b2b266b379fc1db16aaee156`。

### Atomic Crop+TGVF

```text
artifacts/policy/
  PRL-20-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-8step-ws8/
    evaluation/
      PRL20-R0-FROZEN-RP67-TFREE-CROP-TGVF-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1/
        paired-summary.json
        evaluation-complete
```

```text
paired-summary.json SHA256  = e45fa6876facc994b212ed6d7b63eaae221732b4d6a0ede6cb93add88e56fe57
evaluation-complete SHA256  = c92cf45fd5badf9de662848adeef2718b0e88fb8a35357ba2c3998a186b9053f
```

## 12. Final closeout

本 pilot 最重要的成果不是一个孤立最高分，而是一套可以继续信任的实验基线与清晰的
证据边界：Crop 可学，`D` 有用，Frozen RP67 T-free 是当前稳健 TGVF 配方，组合工具
可行，视觉 shaping 有信号但尚未完成属性级证明；同时我们明确识别了 small-batch
noise、single-seed evaluation 和 reward ordering 的限制。

`POLICY-RL-SMALL-BATCH-PILOT-CLOSEOUT-20260814-v1` 至此关闭。后续探索从本报告的
冻结 baseline 出发，以新的实验身份推进。
