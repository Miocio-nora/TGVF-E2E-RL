# PRL13 Crop RL：8-Step 成功 Pilot 报告

日期：2026-08-08  
状态：`PILOT COMPLETE / POSITIVE`  
实验对象：`Qwen3-VL-8B-Instruct` + native Crop + full-model GRPO  
正式比较：同一套 clean Crop protocol 下的 `step 0` 与 `step 8`

## 1. 一句话结论

这次 RL pilot 是当前项目里第一个足够有说服力的正向结果：只训练 8 个 optimizer
step，CoreDev-2511 的七项等权诊断均值从 **55.57% 提升到 60.99%（+5.42
pp）**；在 840 条共同支持的选择题上净增 **50** 道正确答案，准确率从
**56.79% 提升到 62.74%（+5.95 pp）**，成对 McNemar 检验
`p = 4.99e-4`。

更关键的是，模型不是靠“更频繁地乱用工具”获得增益。相反：

- 正常产生最终答案的比例从 **85.45%** 提升到 **94.96%**；
- 触发 6 次 Crop 上限的样本从 **293** 降到 **89**（`-69.6%`）；
- 总 Crop 调用数从 **4,646** 降到 **3,303**（`-28.9%`）；
- 同时 HRBench4K、OCRBench、MathVista、VStar 等外部精度明显提升。

因此这个 pilot 验证的不只是“权重发生了变化”，而是模型学会了更有效地完成
`观察 → 必要时 Crop → 利用新视觉信息 → 停止工具循环 → 作答` 的闭环。

## 2. 先澄清实验身份

配置名称中保留了 `BS256-N16-80STEP`，这是目标配方的历史 run ID；**本次真正
完成并被评测的执行体不是 BS256/80-step，而是 8-step efficacy pilot**：

| 项目 | 实际值 |
|---|---:|
| optimizer steps | 8 |
| prompt batch / step | 16 |
| rollouts / prompt | 16 |
| trajectories / step | 256 |
| 总训练 prompts | 128（无重复） |
| 总训练 trajectories | 2,048 |
| GPU | 8 × B200 |
| FSDP world size | 8 |
| actor micro-batch / GPU | 32 |
| 训练模式 | full model；vision、projector、LLM 均可训练 |

实际执行目录为：

```text
artifacts/policy/
  PRL-13-A-qwen3-instruct-grpo-bs256-n16-native-crop-t1-stratified-80step-gpu0123/
    pilot-bs16-n16-micro32-fused-flex-ws8/
```

W&B：<https://wandb.ai/mio_nora/tgvf-policy-rl/runs/g9j7mkru>

后续报告和实验名应直接使用 `PRL13-A-PILOT-BS16-N16-STEP8-WS8`，避免再把
目标配置名误读为实际训练量。

## 3. 模型与优化配置

| 配置 | 实际值 |
|---|---|
| base model | `Qwen3-VL-8B-Instruct` |
| RL algorithm | GRPO |
| optimizer | AdamW |
| learning rate | `1e-6`，constant，无 warmup |
| PPO epochs | 1 |
| GRPO group size | 16 |
| advantage normalization | group 内按标准差归一化 |
| KL reward coefficient | 0 |
| actor KL loss | 关闭 |
| entropy coefficient | 0 |
| gradient clipping | 1.0 |
| max prompt / response | 8,192 / 20,480 tokens |
| rollout sampling | temperature 1.0，top-p 1.0 |
| max active perceptions | 6 |
| checkpoint | 每 step 保存；最终永久保留 step 8 |

这次采用 full-model 更新，而不是 LoRA。训练图像经过真实 Qwen3-VL
processor，vision encoder、projector 与 language model 均处于可训练路径；step 8
以完整 8-rank FSDP checkpoint 保存，并物化为标准 Hugging Face 权重后交给
stock vLLM 评测，评测中没有加载 LoRA 或项目私有视觉 adapter。

### 3.1 实际吞吐

8 个 step 的 `timing_s/step` 合计约 **5.77 小时**，平均约 **43.3 分钟/step**。
不同 step 因序列长度差异在约 23.7–47.2 分钟之间波动。主要耗时不是 Crop
runtime，而是 old-log-prob 与 full-model actor update；每步保存完整 checkpoint
约 1.9–2.2 分钟。

本次已经验证的显存优化组合为：8-rank FSDP2、关闭 parameter/optimizer/ref
offload、保留 gradient checkpointing、fused kernels、text FlexAttention、vision
SDPA，以及每卡 32 条 actor micro-batch。它把之前的 OOM 配方变成了可连续训练
8 step 的稳定执行体。

## 4. 数据筛选与实际训练数据

### 4.1 T1 的定义

候选样本先由原始 `Qwen3-VL-8B-Instruct` 在 full-image、无工具条件下生成 8 个
rollout。T1 保留“8 次中答对 1–7 次”的样本，即：

- 排除 8/8 答对的过易样本；
- 排除 0/8 答对的过难或异常样本；
- 保留本身有解、但仍然存在策略改进空间的中间难度样本。

本次使用修复 ThinkLite 类型判定后的最终数据：

| source | candidates | T1 retained | retained ratio |
|---|---:|---:|---:|
| VStar | 170,000 | 39,205 | 23.06% |
| ArxivQA | 32,000 | 25,393 | 79.35% |
| ThinkLite | 69,842 | 12,943 | 18.53% |
| **合计** | **271,842** | **77,541** | **28.52%** |

另有 161 条 unresolved 样本没有进入最终 retained pool。最终数据 artifact 为：

```text
artifacts/data/policy_rl/
  T1-04-INSTRUCT-FULL-MIXED-T1-RETAINED-FINAL-v2/
```

关键内容哈希：

```text
content_sha256 = 5ab99622a2698a7c52c45795215fa5c467b741c103827a1a7dbe3800ff052934
samples_sha256 = 06e5b1b9039680111df5ef01f7f969b9cf3d8d0eaefa5774fd8d16169428611a
```

### 4.2 8-step pilot 实际消费的数据

完整 80-step schedule 原计划每 256 prompts 固定为 VStar 120、ArxivQA 77、
ThinkLite 59，并且不放回采样。pilot 将每步 prompt batch 临时缩到 16，因此实际
8 step 共消费 128 个唯一 prompt：

| source | prompts | share |
|---|---:|---:|
| VStar | 56 | 43.75% |
| ArxivQA | 44 | 34.38% |
| ThinkLite | 28 | 21.88% |

每个 prompt 生成 16 条 rollout，共得到 2,048 条训练 trajectory。128 个 GRPO
group 中，116 个 group（90.6%）具有非零组内 reward 差异，说明筛选后的数据与
`n=16` 组合确实提供了充足的相对学习信号。

## 5. Crop protocol

视觉源（VStar、ArxivQA）只暴露一个工具：

```text
image_zoom_in_tool(bbox_2d, label?)
```

执行约束如下：

- `bbox_2d = [x1, y1, x2, y2]` 使用 Qwen3-VL 原图相对的 `0..1000` 坐标；
- runtime 只映射一次到 immutable original RGB pixels；
- 每次成功调用返回一张真正的 crop image，而不是文字描述或预计算 embedding；
- crop observation 以 user/tool-response envelope 送回模型；
- 每条 trajectory 最多 6 次有效 Crop，并预留最后一个 assistant turn 作答；
- ThinkLite 为 image-bearing single-turn 分支，不开放 Crop。

评测使用的 clean prompt 要求最终答案直接以 plain text 输出，不要求
`<answer>...</answer>`。协议身份为：

```text
evaluation_protocol = deepeyes_official_visible_native_crop_v1
prompt_bundle_sha256 = 2b8b6d799ebe4bbfd6b3830344850575141b2293750f857c031a2031426c0dd2
coordinate_mapper = qwen_0_1000_to_source_v1
max_tool_calls = 6
```

## 6. Reward

这次没有加载本地 Qwen3-VL-32B visual judge，也没有启用额外的视觉美学/grounding
judge。训练 reward 是 DeepEyes 风格的 answer、format 与 conditional tool 三部分。

### 6.1 视觉样本：VStar / ArxivQA

令：

- $A \in \{0,1\}$：最终答案是否正确；
- $F \in \{0,-1\}$：格式是否合法；
- $C \in \{0,1\}$：是否至少获得一次成功 Crop observation。

则视觉分支 reward 为：

$$
R_{\mathrm{visual}} = 0.8A + 0.2F + 1.2AC.
$$

因此正确且成功 Crop 的最大 reward 为 2.0；正确但不 Crop 为 0.8；错误答案即使
调用工具也不会得到 conditional-tool reward。这个门控很重要：它没有直接奖励
“调用次数”，而只奖励“最终答对时确实用过工具”。

视觉答案由 text-only `Qwen2.5-72B-Instruct` 判断 candidate answer 与 gold answer
是否语义一致。训练时服务走 OpenRouter，并固定到 DeepInfra provider；它不是
32B VLM，也不读取图像。

### 6.2 ThinkLite

ThinkLite 没有工具奖励：

$$
R_{\mathrm{ThinkLite}} = 1.2A + 0.4F.
$$

先走确定性的数学等价验证，无法确定时再回退到 text-only
`Qwen2.5-72B-Instruct`。8 step 共记录 2,083 次实际 judge 请求、38 次 retry 和 4
次最终失败；失败只将当前样本记零，没有中断整批训练。

## 7. CoreDev-2511 正式结果

### 7.1 评测覆盖

CoreDev-2511 包含 2,511 条任务。当前 Crop runtime 只支持单图，因此真正完成
model inference 的是 **2,240 条单图样本**。其余 271 条多图任务为：

- BLINK：240 条；
- MMMU-Pro：31 条。

它们在官方尺寸表中被确定性填零，不应描述成“模型完成了 2,511 条推理”。下面
同时给出共同单图 support 与官方 zero-padded 分数，避免口径混淆。

step 0 和 step 8 使用完全相同的 task manifest、clean prompt、Crop runtime、坐标
系统和 scorer。step 0 已证明与原始 base HF 权重等价；step 8 是完整 full-model
checkpoint。VLMEvalKit 固定在 commit `7055d301`，需要语义判分的 benchmark 使用
本地 `Qwen2.5-72B-Instruct` judge。scorer parse failure 为 step0 2 条、step8 1 条，
均按 deterministic incorrect 处理。

### 7.2 七项结果

| benchmark / metric | step 0 | step 8 | delta |
|---|---:|---:|---:|
| VStarBench Overall | 78.01 | **81.15** | **+3.14** |
| HRBench4K 4-cycle Average | 53.50 | **70.00** | **+16.50** |
| BLINK single-image support（180） | 57.22 | **60.00** | **+2.78** |
| OCRBench-v2 English | 40.46 | **46.26** | **+5.80** |
| OCRBench-v2 Chinese | 37.45 | **50.07** | **+12.62** |
| MMMU-Pro single-image support（269） | 43.87 | **46.10** | **+2.23** |
| MathVista MINI | 62.67 | **68.33** | **+5.67** |
| MathVerse MINI five-version macro | **54.80** | 53.20 | **-1.60** |

补充官方 zero-padded 口径：

| benchmark | step 0 | step 8 | delta |
|---|---:|---:|---:|
| BLINK full 420 | 24.52 | 25.71 | +1.19 |
| MMMU-Pro full 300 | 39.33 | 41.33 | +2.00 |

七项等权诊断 macro 使用：VStar、HR、BLINK single-image、OCR 英中均值、MMMU
single-image、MathVista、MathVerse five-version macro。结果为：

$$
55.57\% \rightarrow 60.99\%, \qquad \Delta = +5.42\ \text{pp}.
$$

这不是某个 benchmark 的官方总分，而是为了观察跨任务方向一致性定义的诊断
统计。七项中六项提升；唯一回落是 MathVerse。

### 7.3 共同支持选择题的成对统计

VStar 191、HRBench4K 200、BLINK single-image 180、MMMU-Pro single-image 269，
合计 840 条共同支持的 MCQ：

| statistic | step 0 | step 8 | delta |
|---|---:|---:|---:|
| correct | 477 | 527 | +50 |
| accuracy | 56.79% | 62.74% | +5.95 pp |

成对 McNemar exact test：`p = 0.0004994265`。分项中 HRBench4K 是最强的主要
贡献者（107 → 140，`p = 3.76e-5`）；VStar、BLINK 与 MMMU 的单项变化方向为正，
但各自样本量下尚未达到显著。

## 8. 行为机制：模型具体学会了什么

2,240 条共同单图任务的完整 trajectory audit 给出了比准确率更直接的证据：

| behavior | step 0 | step 8 | change |
|---|---:|---:|---:|
| normal final answer | 1,914（85.45%） | 2,127（94.96%） | +213 / +9.51 pp |
| stop at 6-call cap | 293（13.08%） | 89（3.97%） | -204 / -69.6% |
| context limit | 33（1.47%） | 24（1.07%） | -9 |
| samples using Crop | 1,853（82.72%） | 1,791（79.96%） | -62 / -2.77 pp |
| successful Crop calls | 4,646 | 3,303 | -1,343 / -28.9% |
| calls per sample | 2.074 | 1.475 | -28.9% |
| calls per tool-using sample | 2.507 | 1.844 | -26.4% |
| mean sampled tokens | 701.1 | 650.4 | -7.2% |

最有解释力的成对 stop transition 是：

| transition | count |
|---|---:|
| final → final | 1,858 |
| tool-call-cap → final | **250** |
| final → tool-call-cap | 44 |
| tool-call-cap → tool-call-cap | 43 |
| context-limit → final | 19 |

也就是说，RL 主要修复了 base policy 的“已经得到足够信息但仍继续 Crop”问题。
工具使用样本比例只小幅下降，但每个工具样本的调用次数明显下降，最终回答率明显
上升。这与 conditional-tool reward 的设计吻合：reward 鼓励有效工具使用，但没有
鼓励重复调用。

这也排除了最简单的两类 reward hacking 解释：

1. 不是靠把 Crop rate 无限制推高；
2. 不是只靠输出更短来规避失败，因为多个外部 benchmark 的正确率同时提升。

## 9. 相对原始 direct baseline 的位置

项目中还有一份旧的原始 `Qwen3-VL-8B-Instruct` direct、无工具 baseline。它与
本次 Crop 评测在 prompt、采样、工具协议和多图支持上都不完全一致，因此只能作为
上下文，不能替代严格的 step0 对照。

| metric | raw direct（旧协议） | Crop step 0 | Crop step 8 |
|---|---:|---:|---:|
| VStarBench | 50.79 | 78.01 | **81.15** |
| HRBench4K Average | 59.00 | 53.50 | **70.00** |
| OCR 英中 macro | 48.18 | 38.95 | 48.16 |
| MathVista MINI | **74.33** | 62.67 | 68.33 |

正确结论不是“step 8 在所有任务上都支配 raw direct”，而是：在严格同协议的
step0/step8 比较中，RL 带来了清晰而广泛的增益；相对 raw direct，Crop policy
已经在 VStar 与 HR 上明显更强，但 MathVista 仍有恢复空间，OCR 基本回到 direct
水平。

## 10. 这个 pilot 真正验证了什么

### 已验证

1. **当前 native Crop 闭环可学。** 工具调用、真实 crop pixels、多轮 observation、
   policy update、完整 checkpoint 和外部评测已经端到端打通。
2. **Instruct base 是可行起点。** 不需要回到 Thinking base 才能在工具 RL 中取得
   明确增益。
3. **T1 数据足以提供学习信号。** 90.6% 的训练 group 有非零相对 reward；仅 128
   个训练 prompt、2,048 条 rollout 已出现跨 benchmark 增益。
4. **`n=16` 很关键。** 每个 prompt 内有足够的好坏 trajectory 形成 GRPO 对比，
   避免小 rollout 数下的高方差/零信号问题。
5. **full-model recipe 可运行且有效。** vision、projector 和 language 路径都进入
   可训练 full-model 更新，且完整 checkpoint 可由 stock vLLM 读取。
6. **简单 reward 已经足够形成正确方向。** 即使没有 32B VLM judge，answer 主导、
   tool reward 条件化的配方也能让工具行为更健康。
7. **8-step 是有效的小型 scale gate。** 不需要先支付 80-step 或论文级训练成本，
   就能判断 recipe 是否值得扩大。
8. **CoreDev-2511 比早期小 benchmark 更可靠。** 它揭示了广泛提升、MathVerse
   回落以及多图 support 缺口，而不是只给一个平均数。

### 尚未被单独验证

这次验证的是一个**联合 recipe**，不是控制变量消融，因此不能把全部增益单独归因
给 T1、full-model、vision update、`n=16` 或某个 reward coefficient。尤其：

- 实际 prompt batch 是 16，本次没有验证 BS256 的额外收益；
- 没有同预算 LoRA / frozen-vision 对照，不能定量声称 full-model 比 LoRA 高多少；
- 没有 unfiltered-data 对照，不能单独测量 T1 筛选贡献；
- 没有 reward-weight ablation，不能区分 answer reward 与 conditional-tool reward
  各自贡献；
- 只评测了一个 step8 seed/sampling realization，尚无多 seed 置信区间；
- 还没有训练到 step20/45/80，不能外推长期单调增益。

## 11. 两个必须保留的限制

### 11.1 训练 prompt 的 `<answer>` 历史问题

虽然当前代码与正式评测已经切换为 plain final answer，但实际 8-step 训练
trajectory 证明，当时视觉训练 prompt 仍然要求：

```text
<answer>...</answer>
```

这是不希望保留的历史格式，下一次 clean run 必须删除。它不推翻本次 RL 效果：
step0 和 step8 的外部评测都使用相同的 clean、无 `<answer>` 协议，因此权重更新在
clean protocol 下的相对增益仍然成立；但本次不能被称为最终的 clean training
recipe。

### 11.2 多图与采样不确定性

当前 Crop evaluator 只支持单图，BLINK/MMMU 的 271 条多图样本没有真正推理。
此外，正式评测沿用 policy 的 temperature 1.0 单样本采样，并使用绑定到 checkpoint
identity 的确定性 seed；step0/step8 是同任务成对比较，但不是 common-random-number
比较。下一轮应至少增加：

- 一组 greedy 或固定公共 sampling seed 的 step0/step8 复验；
- 2–3 个 evaluation seed 的均值与区间；
- 原生多图 Crop 支持后补齐 BLINK/MMMU full score。

## 12. 下一阶段建议

按信息增益排序：

1. **先冻结 step8 为成功基线。** 不覆盖 checkpoint、评测 trajectory、scorer 输出
   和 W&B run。
2. **做 clean-answer restart。** 删除训练 prompt 中的 `<answer>` 要求，其余 recipe
   不变，先跑 8 step 复现。
3. **补共同 seed / greedy 复验。** 用较低成本确认 +5.42 pp 不依赖单次采样。
4. **在成功 recipe 上继续到 step20。** step8 已经足以过 gate；step20 用于判断
   增益是否继续、平台化或开始过拟合，而不是直接跳到 80。
5. **做一个最小 full-vs-LoRA 控制。** 相同 128 prompts、相同 rollout seed、相同
   reward 和 8 step，只改变 trainable parameter path。
6. **再迁移到 TGVF。** TGVF 的第一版应尽量保持本次已经验证的 GRPO、T1、n=16、
   answer-dominant reward 与评测协议，仅把 Crop observation 换成 RP66/RP67 的 D，
   避免一次改变过多变量。

## 13. 关键 artifact 索引

```text
# 训练 trajectories 与 step8 checkpoint
artifacts/policy/PRL-13-A-qwen3-instruct-grpo-bs256-n16-native-crop-t1-stratified-80step-gpu0123/
  pilot-bs16-n16-micro32-fused-flex-ws8/

# T1 final dataset
artifacts/data/policy_rl/T1-04-INSTRUCT-FULL-MIXED-T1-RETAINED-FINAL-v2/

# step0 / step8 CoreDev-2511 paired evaluation
artifacts/evaluation/PRL13-A-CoreDev2511-clean-no-answer-paired-mem080-v1/

# official summaries
.../step0/scoring/coredev-official-v2/coredev-2511-eval-summary.json
.../step8/scoring/coredev-official-v2/coredev-2511-eval-summary.json
```

关键权重身份：

```text
step0 weights_sha256 = ad897b7ec2f8f2c0046346b74c003827defc7847c9c099a26cd8f9c8ee237932
step8 weights_sha256 = d2cc3d47e6dd8bb08b6ef74c31f40239b082aadd641e40fca13057e59e4308fd
run identity_sha256  = 20821298646c4042cf0411038fd44a9863f3aa8ad00615f3fe2a5aed28e5061c
```

## 14. 最终判断

PRL13 step8 不是“训练终于没报错”意义上的成功，而是一个明确通过 efficacy gate
的成功 pilot：**正确筛选
的 Instruct T1 数据、n=16 GRPO、full-model 可训练视觉路径、native Crop 和
answer-gated tool reward 的联合配方，能够在很小的 8-step 预算内显著改善外部
任务精度，并把工具行为从重复 Crop/无法停止，转变为更少调用、更高完成率的健康
策略。**

这已经足以作为下一轮 clean Crop scaling 和 TGVF RL 的共同基线；它不代表
BS256、20/80-step 或论文规模训练已经完成，也不代表长期 scaling 结论已经得到
验证。
