# BS16 Crop、TGVF 与 Crop+TGVF 对比资料页

日期：2026-08-20（Asia/Tokyo）
用途：聚合论文实验分析所需的 BS16 结果、训练配置、时间与证据边界。

## 1. 范围与口径

本文只汇总以下四条历史 BS16 policy-RL 线路：

| 本文简称 | 实验身份 | 说明 |
|---|---|---|
| Crop clean-final | PRL14-A | native Crop，原 DeepEyes-style reward |
| Crop T-free | PRL21-R0 | native Crop，改用与 TGVF 主线相同的 T-free reward 主体 |
| TGVF | PRL22-A Teacher25 | pure TGVF，Frozen RP67，当前 BS16 最佳 S16 |
| Crop+TGVF | PRL22-B Teacher25 | Atomic Crop+TGVF，Frozen RP67，当前 BS16 最佳 S16 |

`Macro*` 是 VStar、HRBench、BLINK single-image-180、OCR EN/CN mean、MMMU
single-image-269、MathVista 和 MathVerse five-version macro 的七项等权均值。
OCR English 与 Chinese 在表中分别展示，但只以二者均值计入一次 `Macro*`。

> 重要版本说明：这里的 PRL21/PRL22 是历史 BS16 T-free 结果，协议/工具错误罚分为
> `-1`；Crop clean-final 使用另一套原 DeepEyes-style 公式。它们都早于 PRL24 当前
> 采用的 FMT2 `-2`，因此不能写成 FMT2 的实验证据。

## 2. 完整结果表

所有数值单位为 `%`。`—` 表示未评测，不能用其他工具线路的 S0 补造。

| 指标 | Crop clean S0 | Crop clean-final S16 | Crop T-free S16 | TGVF 对应 S0 | 最佳 TGVF S16 | Crop+TGVF 对应 S0 | 最佳 Crop+TGVF S16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| VStar | 78.01 | 79.06 | 75.92 | 62.83 | 69.11 | — | 72.77 |
| HRBench Average/all | 53.50 | 60.00 | 73.50 | 58.50 | 69.50 | — | 70.00 |
| BLINK single-image-180 | 57.22 | 57.78 | 62.78 | 62.78 | 60.56 | — | 68.33 |
| OCR English | 40.46 | 46.80 | 49.79 | 46.20 | 45.83 | — | 53.05 |
| OCR Chinese | 37.45 | 51.11 | 57.66 | 40.87 | 36.23 | — | 51.84 |
| OCR EN/CN mean | 38.95 | 48.95 | 53.72 | 43.53 | 41.03 | — | 52.45 |
| MMMU single-image-269 | 43.87 | 50.93 | 45.35 | 50.19 | 50.19 | — | 45.72 |
| MathVista MINI | 62.67 | 65.33 | 65.33 | 68.00 | 73.33 | — | 71.00 |
| MathVerse five-version macro | 54.80 | 54.80 | 51.00 | 53.40 | 56.00 | — | 57.20 |
| **Macro*** | **55.5742** | **59.5502** | **61.0862** | **57.0320** | **59.9590** | **—** | **62.4974** |

## 3. 训练变化

| 线路 | S0 | S8 | S16 | 可报告的变化 |
|---|---:|---:|---:|---|
| Crop clean-final | 55.5742 | 59.7161 | 59.5502 | S16−S0 `+3.9760 pp`；S8/S16 基本平台 |
| Crop T-free PRL21 | 55.5742† | 61.1032 | 61.0862 | S16−复用 S0 `+5.5120 pp`；S16−S8 `-0.0170 pp` |
| TGVF Teacher25 | 57.0320 | 58.4655 | 59.9590 | paired S16−S0 `+2.9270 pp` |
| Crop+TGVF Teacher25 | — | 62.3719 | 62.4974 | S16−S8 `+0.1255 pp`；没有 S0 |

† PRL21 没有重新推理 S0，只复用了协议相同的历史 Crop clean S0。因此
`+5.5120 pp` 是 pilot 上下文，不是 paired causal delta。

## 4. 训练配置

### 4.1 共同配置

| 字段 | 配置 |
|---|---|
| Base policy | `Qwen3-VL-8B-Instruct` |
| RL algorithm | GRPO |
| Policy update | full Qwen；vision encoder、merger/projector、language model 均更新 |
| Prompt batch | BS16 |
| Rollouts | n16，即每步 `16 × 16 = 256` trajectories |
| Horizon | S0→S16；S8、S16 为主要保留与评测点 |
| Distributed | 8 × B200，FSDP2，world8，GA1 |
| Optimizer | AdamW，constant LR `1e-6`，无 warmup，gradient clipping `1.0` |
| PPO / KL | PPO epoch 1；KL reward `0`；actor KL loss off |
| Sampling | temperature `1.0`，top-p `1.0` |
| Maximum length | prompt `8,192` tokens；response `20,480` tokens |

### 4.2 线路差异

| 配置 | Crop clean-final | Crop T-free PRL21 | TGVF PRL22-A | Crop+TGVF PRL22-B |
|---|---|---|---|---|
| Tool | `image_zoom_in_tool` | `image_zoom_in_tool` | `tgvf_focus_tool` | `tgvf_crop_tool` |
| Observation | 原图上的 RGB crop | 原图上的 RGB crop | target-conditioned latent `D` + D-DeepStack | crop 经 Qwen vision 与 RP67 后的 latent `D` |
| Representation | 无额外 Adapter | 无额外 Adapter | RP67 Step-2000，frozen | RP67 Step-2000，frozen |
| Policy data | T1 retained，no Teacher | T1 retained，no Teacher | Teacher25 | Teacher25 |
| Reward | DeepEyes-style conditional Crop reward | T-free | T-free | T-free |
| Evaluation RNG | legacy-RNG | legacy-RNG | paired-seed-v1 | 自己的 paired-seed-v1 block |

Teacher25 的确定性 schedule 共 `20,480` 个 prompts，seed 42、无放回；每个 BS16
group 固定为 12 条 existing data 和 4 条 Teacher data，即 75%/25%。Teacher 数据来自
ChartQA、DocVQA、TextOCR、TextVQA 与 Visual Genome。

### 4.3 Reward 公式

Crop clean-final 的历史视觉样本 reward：

```text
R_crop = 0.8 × AnswerCorrect
       + 0.2 × FormatScore
       + 1.2 × AnswerCorrect × HasSuccessfulCrop
```

PRL21 Crop、PRL22 TGVF 与 PRL22 Crop+TGVF 的历史 T-free reward 主体：

```text
R_tfree = 2 × AnswerCorrect
        − 0.05 × max(0, ToolCallCount − 1)
        − 1 × 1[ProtocolOrToolError]
```

该 T-free 配方关闭 tool utility `T`、focus、grounding 与 positive-crop bonus。
“Reward 平齐的 Crop”只表示 PRL21 与 TGVF/Crop+TGVF 使用相同的 T-free scalar
reward 主体；PRL21 没有 Teacher25，且 prompt/tool/RNG block 仍不同，因此不是 fully
matched control。

## 5. 训练时间记录

下表只统计训练，不包含后续完整 CoreDev-2511 推理与评分时间。

| Run | 训练日期/时间（JST） | 已记录训练时间 | 说明 |
|---|---|---:|---|
| Crop clean-final PRL14-A | 2026-08-09 13:49:35 完成 | 未记录可靠连续 wall time | `completion.json` 只有完成时间；中间存在恢复，不能从文件时间反推纯训练耗时 |
| Crop T-free PRL21-R0 | 2026-08-14 21:39:59 → 2026-08-15 06:25:50 | **10 h 34 min 50 s** | 16 个成功 optimizer step 的 `timing_s/step` 合计 `38,090.230 s`；平均 **39 min 41 s/step**。completion 的 8 h 45 min 51 s 只覆盖最终 resume lifecycle，不能当成完整 16-step wall time |
| TGVF PRL22-A Teacher25 | 2026-08-16 00:20:08 主运行 → 03:16:11 最终 metrics | **2 h 46 min 55 s** | 16 步 `end_to_end_step_seconds` 合计；平均 10 min 26 s/step；artifact 时间跨度约 2 h 56 min |
| Crop+TGVF PRL22-B Teacher25 | 2026-08-16 04:08:09 → 08:01:49 最终 metrics | **3 h 44 min 24 s** | 16 步 `end_to_end_step_seconds` 合计；平均 14 min 02 s/step；artifact 时间跨度约 3 h 54 min |

在相同 BS16 × n16、world8、16-step 配置下，Atomic Crop+TGVF 比 pure TGVF 的
累计 end-to-end step 时间多约 `3,449 s`，即 **57 min 29 s**（约 `34.4%`）。这是
工具执行与序列形态共同作用下的系统成本观察，不应解释为方法准确率差异的原因。

PRL21 Crop 经多次恢复完成。完整训练成本必须按各次成功 step 的计时求和；从最终
`completion.json` 的 created/completed 时间相减所得 8 h 45 min 51 s 漏掉了早期成功
step，今后的容量规划不得再引用该值。按当前历史均值线性外推，单条 BS16 Crop 80-step
约需 52 h 54 min；这只描述 PRL21 旧执行体，不再作为 PRL25 exact-Crop ETA。

2026-08-21 的 PRL25-B 对齐版 1-step canary（BS4 × n2、world4）完整通过，端到端
`262.27 s`，其中 publication 前 `162.04 s`、full-Qwen sync `4.09 s`、checkpoint
`96.14 s`。它确认旧纯 Crop 的额外 logprob/replay 与存盘路径不是方法本身不可避免的
成本，但样本规模只有正式 BS16 × n16 的 1/32，因此只能作为功能与成本分解证据；正式
80-step ETA 必须等待 PRL25-B 的首个 BS16 × n16 step 完整发布。

## 6. 简短分析

1. **Crop policy-RL 本身可以学习。** 原 DeepEyes-style Crop 从 S0 到 S16 提升
   `+3.9760 pp`；改用 T-free 后，PRL21 S16 达到 `61.0862`，比原 Crop S16 高
   `+1.5360 pp`，但该跨 run 差值不是 paired reward ablation。
2. **纯 TGVF 更偏向 HRBench 与数学能力。** 相对 Crop clean-final S16，TGVF S16
   的 HRBench 为 `+9.50 pp`、MathVista 为 `+8.00 pp`，但 VStar 为 `-9.95 pp`、
   OCR mean 为 `-7.92 pp`。
3. **Atomic Crop+TGVF 的整体分数最高。** 其 S16 Macro* 为 `62.4974`，比纯 TGVF
   S16 高 `+2.5384 pp`；主要差异来自 BLINK（`+7.77 pp`）和 OCR mean
   （`+11.42 pp`）。它仍低于 Crop clean-final S16 的 VStar（`-6.29 pp`）和 MMMU
   （`-5.21 pp`）。
4. **当前证据支持“互补能力”，不支持严格 synergy。** Crop、TGVF 与组合工具使用
   不同 prompt/tool/RNG blocks，且组合线路没有 S0，因此只能做描述性横向比较。

## 7. 论文措辞边界

可以安全使用：

- “在单 seed 的 BS16 pilot 中，Atomic Crop+TGVF 在 CoreDev-2511 上取得最高的
  S16 Macro*（62.4974）。”
- “纯 TGVF 与 Crop 呈现不同的能力侧重，而组合工具在 BLINK 与 OCR 上恢复了纯
  TGVF 的主要短板。”
- “Teacher25 在 pure TGVF 和 Atomic Crop+TGVF 的 matched-recipe S16 对照中分别
  带来 `+1.7595 pp` 和 `+1.5434 pp`。”

不应使用：

- “Crop+TGVF 相对自己的 S0 提升了多少”：该 S0 未评测。
- “组合工具带来严格的 `+2.9472 pp` synergy”：这是相对 Crop clean-final S16 的
  跨协议描述性差值。
- “PRL21 证明 T-free reward 因果优于 DeepEyes reward”：两者不是 paired reward
  ablation。
- “这些结果验证了 FMT2”：PRL21/PRL22 T-free 使用 `-1` 错误罚分，Crop clean-final
  使用另一套原 DeepEyes-style 公式；它们都不是当前 PRL24 FMT2 `-2`。

## 8. 第三期（PRL25）如何使用这些历史结果

历史结果显示 Crop clean、Crop T-free 和 Crop+TGVF 都在 S8→S16 附近接近平台，但
16-step 曲线不足以区分“已经达到能力上限”和“需要更长优化期”。因此第三期把主终点
延长到 S80，而不是继续从单个 S16 横向差值下结论。

第三期统一采用 `BS16 × n16`、Teacher25、FMT2 `-2`、相同 S0 和 80-step horizon，五条
fresh arm 为：DeepEyes-style conditional Crop、自研 T-free Crop、T-free TGVF、T-free
Atomic Crop+TGVF，以及 T-free + Focus/Target + Grounding 的 TGVF。每条都重新训练；
PRL21/22 checkpoint 和 PRL24-D S1 均不作为初始化。

本页的历史数值仍可作为能力与运行时间锚点，但 PRL21/22 使用 FMT1 `-1`，Crop
clean-final 的 reward/data 又不同，不能直接作为第三期的 matched FMT2 对照。正式矩阵、
比较边界、checkpoint 和评测规则见
[PRL25 第三期 BS16 Teacher25 80-step 计划](PRL25_BS16_TEACHER25_80STEP_PHASE3_PLAN_20260820.md)。

## 9. 来源

- [CoreDev-2511 measurement contract 与 canonical 总表](POLICY_RL_COREDEV2511_MEASUREMENT_CONTRACT_AND_BASELINES_20260812.md)
- [BS16 small-batch pilot closeout](POLICY_RL_SMALL_BATCH_PILOT_CLOSEOUT_20260814.md)
- [PRL21 Crop T-free 结果与评测事件](PRL21_CROP_TFREE_16STEP_RESULTS_AND_EVALUATION_INCIDENT_20260815.md)
- [PRL22 Teacher25 policy-data ablation](PRL22_TEACHER25_POLICY_DATA_ABLATION_RESULTS_20260816.md)
- [PRL14/历史 Crop 配置对照](POLICY_RL_PRIMARY_BASELINE_20260810.md)
- [PRL25 第三期 BS16 Teacher25 80-step 计划](PRL25_BS16_TEACHER25_80STEP_PHASE3_PLAN_20260820.md)

对应训练 artifact：

```text
artifacts/policy/PRL-14-A-qwen3-instruct-grpo-bs16-n16-native-crop-t1-cleanfinal-16step-ws8/
artifacts/policy/PRL-21-R0-qwen3-instruct-full-crop-bs16-n16-tfree-16step-ws8/
artifacts/policy/PRL-22-A-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher25-8step-ws8/
artifacts/policy/PRL-22-B-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-teacher25-8step-ws8/
```
