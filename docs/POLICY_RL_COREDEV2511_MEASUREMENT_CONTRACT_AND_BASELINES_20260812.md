# Policy RL CoreDev-2511 统一测量标准与主基线

日期：2026-08-12

结果更新：2026-08-13

状态：`PRIMARY MEASUREMENT CONTRACT / FROZEN V1`

Contract ID：`POLICY-RL-COREDEV2511-MEASUREMENT-20260812-v1`

适用范围：后续 Qwen3-VL-8B-Instruct Crop / TGVF policy-RL 的 Step 0、训练中间点与最终 checkpoint 对比。

本文冻结两件事：

1. CoreDev-2511 的统一测量与聚合口径；
2. canonical 大表的汇报结构；实验结果持续按同一契约追加，当前更新至 2026-08-13。

本文只替代旧文档中的 headline 聚合值，不否定旧文档记录的模型、prompt、checkpoint、训练配置与 artifact 身份。特别是 `docs/POLICY_RL_PRIMARY_BASELINE_20260810.md` 中使用 HRBench cycle 0 和 OCR Chinese-only 得到的旧均值，不再作为主汇报值。

Crop Step 0/8/16 的同标准 `legacy-RNG` 结果位于第 3.1 节；RP67 T-free Step 0/8/16 `paired-seed-v1` 结果详记在第 7 节，joint/unfrozen RP67 Step 8/16 结果详记在第 8 节；两种 RP67 Adapter 更新方式的横向 paired 总表位于第 3.2 节。这些结果不修改本文件冻结的 benchmark、scorer、prompt、sampling 或聚合契约；第 3.1 节的 legacy-RNG 结果与第 3.2 节的 paired-seed-v1 结果必须按 RNG 身份分别引用。

## 0. 当前决策摘要

在已完成的对照范围内，后续 TGVF policy-RL 的默认配方定为：

```text
Stage 1 Adapter = RP67
Adapter during RL = frozen
policy during RL = full Qwen3-VL-8B-Instruct
                   (vision encoder + merger + language model all trainable)
reward = T-free
         answer correctness + protocol/tool-error penalty
         + repeated-call penalty
disabled reward = T/tool-utility, focus, grounding
```

三条主要轨迹的一页摘要如下。Delta 只在同一行、同一 RNG block 内计算；Crop 与 RP67 之间不做严格 paired delta。

| 线路 / RNG block | Step 0 | Step 8 | Step 16 | Step 16 − Step 0 |
|---|---:|---:|---:|---:|
| Crop clean → clean-final / legacy | 55.5742 | **59.7161** | 59.5502 | **+3.9760 pp** |
| RP67 T-free Frozen / paired | 57.0320 | 56.1964 | **58.1996** | **+1.1675 pp** |
| RP67 T-free Joint / paired | 57.0320 | 56.2035 | 56.5283 | -0.5038 pp |

| 决策 | 当前证据 | 证据边界 |
|---|---|---|
| **Frozen Adapter** | 严格 paired 下，Frozen S16 `58.1996`，Joint S16 `56.5283`，Frozen 高 **`1.6713 pp`**；Joint S16 还低于 common S0 `57.0320` | 这是目前最强、最直接的因果对照；支持 RL 时冻结 RP67，只更新 full Qwen policy |
| **T-free reward** | legacy 筛选中，`+T` 为 `57.38 → 56.30`，T-free 为 `56.37 → 57.28`；随后 T-free paired S16 相对 common S0 为 **`+1.1675 pp`** | `+T` 与 T-free 尚不是共享随机流的严格 reward ablation；因此 T-free 是当前最受支持的默认，不写成已经统计证明的独立因果结论 |
| **RP67** | RP67 是当前进入 paired policy-RL 主线并取得最佳 TGVF checkpoint 的 Stage 1 版本；其 image-axis 目标提供结构动机 | 本文没有提供 RP66 vs RP67 同 `paired-seed` 严格结果；因此 RP67 是当前工程默认，不是本表已单独证明优于 RP66 的结论 |

因此，本文后续出现“当前 TGVF 默认线”时，均指 **RP67 + T-free + Frozen Adapter**。解冻 Adapter 不再作为默认设定；若再测，必须作为显式 ablation。

## 1. 今后的 headline 口径

### 1.1 数据与 scorer

| 字段 | 固定值 |
|---|---|
| suite | CoreDev-2511 |
| task manifest | `CoreDev2511-official-visible-v1/tasks.jsonl` |
| task manifest SHA256 | `3f69119d24867c3f3210c8b01eb71304247725ddaf9ca983d2b41c2885403cbc` |
| official rows | 2,511 |
| 工具 runtime 实际推理 | 2,240 条单图 |
| held multi-image | 271 条：BLINK 240；MMMU-Pro 31 |
| VLMEvalKit commit | `7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f` |
| semantic judge | 本地 `Qwen2.5-72B-Instruct`，仅用于对应 benchmark 的语义判分 |
| OCR | VLMEvalKit rule-based scorer，不调用 semantic judge |
| 单样本解析失败 | `deterministic_incorrect`，同时报告数量 |
| 系统/服务失败 | 只重试或恢复受影响样本；未补齐前不得发布 headline |

一个异常样本不得导致整套评测报废；但系统错误也不得被静默计成模型错误。

### 1.2 七个等权分量

所有分量先换算为百分数，再用未四舍五入的值聚合；表格最后显示两位小数。

| # | 分量 | 唯一合法取值方式 |
|---:|---|---|
| 1 | VStarBench | `Overall` |
| 2 | HRBench4K | 从 `*_HRBench4K_acc.csv` 读取 `cycle=Average, type=all`；禁止使用 cycle 0 或扁平 JSON 的序号字段 |
| 3 | BLINK | 共同单图支持集，`n=180` |
| 4 | OCRBench v2 | `(English Overall + Chinese Overall) / 2`；EN、CN 必须同时展示，但在 Macro 中合计只占一个分量 |
| 5 | MMMU-Pro-10c | 共同单图支持集，`n=269` |
| 6 | MathVista MINI | `Task&Skill=Overall\|acc` |
| 7 | MathVerse MINI | Text Dominant、Text Lite、Vision Dominant、Vision Intensive、Vision Only 五个 `Overall` 的等权均值 |

统一诊断均值定义为：

```text
Macro* = mean(VStar, HR-Average, BLINK-single, OCR-EN/CN-mean,
              MMMU-single, MathVista, MathVerse-five-version)
```

`Macro*` 是跨 benchmark 的非官方诊断均值，不是任何 benchmark 的官方总分。

### 1.3 明确禁止混入 headline 的数值

- HRBench `cycle=0, type=all`；
- OCR Chinese-only 或 English-only；
- BLINK full-420 / MMMU full-300 的多图 zero-padding 分数；
- 未使用相同 prompt/runtime/sampling 的旧评测；
- 带历史 `<answer>...</answer>` 输出协议的 checkpoint；
- 2026-08-12 的未完成 `temperature=0` greedy stress run；
- 旧 PRL13 Step 8，不能替代当前 PRL14 clean-final Crop Step 8。

BLINK full-420 和 MMMU full-300 可以作为辅助表报告，但必须标注 `zero-padded / non-headline`。

## 2. 推理协议

### 2.1 工具 arm 的主能力评测

后续 Crop/TGVF checkpoint 的正式主评测固定使用训练/正常部署分布：

| 字段 | 固定值 |
|---|---:|
| temperature | `1.0` |
| do_sample | `true` |
| top_p | `1.0` |
| top_k | `-1` |
| min_p | `0.0` |
| repetition penalty | `1.0` |
| presence / frequency penalty | `0.0 / 0.0` |
| cumulative max response | `20,480` tokens |
| max model length | `32,768` |
| maximum tool calls | `6` |
| generations per sample | `1` |
| final answer | plain text；无 `<answer>` wrapper |

同一实验的 Step 0 / Step N 必须使用相同 task manifest、prompt、tool runtime、sampling、scorer 和 judge。只允许 checkpoint 权重与明确声明的实验变量不同。

Original arm 是 raw direct 端到端参考，历史配置使用 `temperature=1`、`max_new_tokens=8192`、`max_model_len=65536`，且没有工具 prompt。因此 Original 不是工具 arm 的严格 paired control；它只能回答“原始 Instruct 模型的 direct 能力是多少”。

### 2.2 `temperature=1` 的随机性与 paired seed

旧版 content-addressed RNG 把 `evaluation_id` 纳入 `trajectory_id`。因此即使权重、prompt 与 task 完全相同，只要换 evaluation ID，就会换采样随机流。Crop S0/S8/S16 和第 3.1 节的 RP67 历史评测都属于这个 `legacy-RNG` 区块。

这不是可以忽略的理论问题：RP67 R1 与 R2 的 Step 0 权重完全相同，但 Macro* 分别为 `57.38` 和 `56.37`，观测差为 `1.01 pp`。

`paired-seed-v1` 已在 RP67 Frozen/Joint 正式评测中实现并验证。它使用共享 RNG namespace：

```text
seed = H(master_seed, task_manifest_sha, protocol_sha,
         sample_id, rollout_index, assistant_turn_index)
```

seed 身份必须排除 `evaluation_id`、arm 名、optimizer step 和 checkpoint hash，使同一题在 Step 0 / Step 8 / Step 16 使用同一初始随机流。后续新的正式 checkpoint/reward/Adapter 对照必须默认使用该 paired 协议，除非明确标记为 stress/diagnostic。

已有 legacy artifact 不会因新功能落地而追溯变成 paired。在 legacy 区块内，小于或约等于 `1.01 pp` 的单次变化一律只写作“趋势”。Paired 结果可以计算同块 delta，但仍只有一个 seed、每题一次采样，不能据此宣称已经统计确认。

### 2.3 `temperature=0` 的定位

纯 greedy 不再用于主准确率。2026-08-12 的 partial run 已证明它显著改变模型失败模式：

| arm | 已完成 | max-token 循环 | 比例 |
|---|---:|---:|---:|
| RP67 T-free Step 0 | 1,507 | 124 | 8.23% |
| RP67 T-free Step 8 | 1,175 | 137 | 11.66% |

几乎全部触顶样本都是答案句、伪工具调用或 OCR 坐标的机械重复，而不是有效长推理。旧 `temperature=1` 全量触顶率仅为 Step 0 `0.94%`、Step 8 `1.25%`。

因此该 run 的身份固定为：

```text
ABORTED / GREEDY-STABILITY STRESS DIAGNOSTIC / NOT ACCURACY EVIDENCE
```

其 partial artifact 可以用于研究 termination pathology，但不得进入本文件主表。

## 3. Canonical 大表（更新至 2026-08-13）

### 3.1 Legacy-RNG 历史总表

所有数字单位为 `%`。OCR mean 是 EN/CN 的均值；Macro* 只把 OCR mean 计入一次。

注意：本节所有工具 arm 都保留自历史 `legacy-RNG` 评测，seed 会随 evaluation ID 改变。它们共享本文的 benchmark/scorer/Macro* 测量标准，但不共享 paired 随机流。它们不能被无标签地替换为第 7 节的 `paired-seed-v1` 数值，也不能与 paired 结果跨块计算 delta。Original 还使用 direct prompt，只是端到端参考。

| benchmark | Original | Crop clean S0 | Crop clean-final S8 | Crop clean-final S16 | RP67 +T S0 | RP67 +T S8 | RP67 T-free S0 | RP67 T-free S8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VStarBench Overall | 50.79 | 78.01 | 76.96 | 79.06 | 66.49 | 58.64 | 64.92 | 65.45 |
| HRBench Average / all | 59.00 | 53.50 | 62.50 | 60.00 | 59.50 | 60.00 | 58.00 | 62.50 |
| BLINK single-image（180） | 65.56 | 57.22 | 60.00 | 57.78 | 59.44 | 63.89 | 63.33 | 64.44 |
| OCRBench v2 English | 49.89 | 40.46 | 47.39 | 46.80 | 46.12 | 44.99 | 45.47 | 44.54 |
| OCRBench v2 Chinese | 46.48 | 37.45 | 51.21 | 51.11 | 34.19 | 37.66 | 36.35 | 37.83 |
| OCR EN/CN mean | 48.19 | 38.95 | 49.30 | 48.95 | 40.16 | 41.33 | 40.91 | 41.19 |
| MMMU-Pro single-image（269） | 39.03 | 43.87 | 47.58 | 50.93 | 48.33 | 47.58 | 45.35 | 48.33 |
| MathVista MINI | 74.33 | 62.67 | 67.67 | 65.33 | 73.33 | 65.67 | 68.67 | 65.67 |
| MathVerse five-version macro | 50.60 | 54.80 | 54.00 | 54.80 | 54.40 | 57.00 | 53.40 | 53.40 |
| **Macro\*** | **55.36** | **55.5742** | **59.7161** | **59.5502** | **57.38** | **56.30** | **56.37** | **57.28** |

对应的 RL delta：

| 线路 | Step 0 | Step 8 | Step 16 | 同 legacy 块的观测变化 | 当前解释 |
|---|---:|---:|---:|---:|---|
| Crop clean-final | 55.5742 | 59.7161 | 59.5502 | S8−S0 **`+4.1419 pp`**；S16−S0 **`+3.9760 pp`**；S16−S8 `-0.1659 pp` | 明显正向 pilot；S8/S16 是平台期，不应将 `-0.17 pp` 解读为真实退化 |
| RP67 +T | 57.38 | 56.30 | — | S8−S0 `-1.08 pp` | 单次负向趋势；幅度接近旧 RNG 波动参照，不能单独定性 |
| RP67 T-free | 56.37 | 57.28 | — | S8−S0 `+0.91 pp` | 单次正向趋势；本块单独不足以证明 reward 优势，需结合第 3.2 节 paired 结果 |

Crop S16 的 BLINK 和 MMMU-Pro 分别为 `104/180` 和 `137/269`。Crop 的 S0/S8/S16 共享同一 CoreDev manifest、Crop prompt/runtime、sampling 参数与 scorer 契约，但 evaluation ID 不同，因此仍是 legacy 同标准比较，不是 common-random-numbers 严格 paired 对照。

### 3.2 Paired-seed-v1 RP67 冻结 / 联合更新总表

下表是当前 RP67 线路唯一允许直接计算 checkpoint delta 的总表。五个 checkpoint 使用相同 CoreDev-2511 manifest、推理协议、`temperature=1`、`master_seed=42` 和 common-random-numbers seed namespace。Joint 实验没有重复评测 Step 0；其初始化与 frozen 实验的 paired Step 0 相同，因此共用 `57.0320` 作为起点。

所有数字单位为 `%`；Macro* 使用未四舍五入的七个分量计算。

| benchmark | Common paired S0 | Frozen S8 | Frozen S16 | Joint S8 | Joint S16 |
|---|---:|---:|---:|---:|---:|
| VStarBench Overall | 62.83 | 65.45 | 64.92 | 64.40 | 57.59 |
| HRBench Average / all | 58.50 | 60.00 | 64.50 | 61.50 | 67.00 |
| BLINK single-image（180） | 62.78 | 60.56 | 63.33 | 64.44 | 62.22 |
| OCRBench v2 English | 46.20 | 44.33 | 43.83 | 42.86 | 42.27 |
| OCRBench v2 Chinese | 40.87 | 34.45 | 37.95 | 29.87 | 30.29 |
| OCR EN/CN mean | 43.53 | 39.39 | 40.89 | 36.36 | 36.28 |
| MMMU-Pro single-image（269） | 50.19 | 47.58 | 47.96 | 47.58 | 49.07 |
| MathVista MINI | 68.00 | 68.00 | 70.00 | 67.33 | 68.33 |
| MathVerse five-version macro | 53.40 | 52.40 | 55.80 | 51.80 | 55.20 |
| **Macro\*** | **57.0320** | **56.1964** | **58.1996** | **56.2035** | **56.5283** |

对应的 paired delta：

| Adapter 模式 | Step 8 − S0 | Step 16 − S0 | Step 16 − Step 8 | 当前结论 |
|---|---:|---:|---:|---|
| Frozen RP67 | -0.8356 pp | **+1.1675 pp** | **+2.0032 pp** | 当前最佳；支持继续验证正向信号 |
| Joint / unfrozen RP67 | -0.8286 pp | -0.5038 pp | +0.3248 pp | 不支持解冻 Adapter 带来总体增益 |

Joint S8 与 Frozen S8 几乎相同（`+0.0071 pp`），但 Joint S16 比 Frozen S16 低 `1.6713 pp`。因此当前证据支持冻结 RP67 Adapter，让 RL 只更新 full Qwen policy（包括 vision encoder、merger 和 language model）；Joint S16 的 HR、MathVerse 增长伴随 VStar、OCR 等回落，属于能力重分配而非总体增强。

## 4. Artifact 来源

### Original direct reference

```text
artifacts/evaluation/
  PRL-04-R2-raw-instruct-coredev2511-gpu4567-r4/
```

### Crop clean Step 0

```text
artifacts/evaluation/
  PRL13-A-CoreDev2511-clean-no-answer-paired-mem080-v1/
    step0/scoring/coredev-official-v2/
```

### Crop clean-final Step 8 / Step 16

```text
artifacts/evaluation/
  PRL14-A-CoreDev2511-cleanfinal-step0-step8-step16-v1/
    step8/scoring/coredev-official-v2/
    step16/scoring/coredev-official-v2/
```

该 PRL14 CoreDev artifact 本身只实测 Step 8/16；表中 Step 0 来自上一个 PRL13 clean Step 0 artifact。三者的 policy weights SHA256 分别为：

| checkpoint | policy weights SHA256 |
|---|---|
| Crop Step 0 | `ad897b7ec2f8f2c0046346b74c003827defc7847c9c099a26cd8f9c8ee237932` |
| Crop Step 8 | `54bf8864114b4b2b80c7603349d02425681a584fe7e4c6ea2c2b3d17fd4ae25d` |
| Crop Step 16 | `50f5d9dd7ecdbf8d9baf46c00c13b1c3719de37b09f5aa91c40aabc758e06beb` |

### RP67 +T R1 Step 0 / Step 8

```text
artifacts/policy/
  PRL-17-R1-qwen3-instruct-full-frozen-rp67-bs16-n16-t1-shaped-novisual-8step-ws8/
    evaluation/PRL17-R1-FROZEN-RP67-COREDEV2511-STEP0-STEP8-SAME-PROTOCOL-V1/
      paired-summary.json
```

### RP67 T-free R2 Step 0 / Step 8

```text
artifacts/policy/
  PRL-17-R2-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-novisual-8step-ws8/
    evaluation/PRL17-R2-FROZEN-RP67-TFREE-COREDEV2511-STEP0-STEP8-SAME-PROTOCOL-V1/
      paired-summary.json
```

### RP67 T-free R2 Step 0 / Step 8 / Step 16 paired-seed-v1

```text
artifacts/policy/
  PRL-17-R2-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-novisual-8step-ws8/
    evaluation/PRL17-R2-FROZEN-RP67-TFREE-COREDEV2511-STEP0-STEP8-STEP16-PAIRED-SEED-V1/
      paired-summary.json
```

该 summary 的 SHA256 为 `bf90a99f52f1943509fa83b8c377c959d32699e5127021ea1b09c49941119176`。

### RP67 T-free Joint Step 8 / Step 16 paired-seed-v1

```text
artifacts/policy/
  PRL-18-R0-qwen3-instruct-full-joint-rp67-bs16-n16-tfree-novisual-8step-ws8/
    evaluation/PRL18-R0-JOINT-RP67-TFREE-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1/
      paired-summary.json
```

R1/R2 Step 0 的共同身份：

| 字段 | SHA256 |
|---|---|
| combined policy weights | `3dd3a76462033a9fb0eaf11db61c3057645ec400676f552fa2b045df673cbed2` |
| Qwen tree | `73a9823eaa1d54f8621ef1cc11bacfe19e1ab13a396c063837f73417caa5603b` |
| RP67 state | `f223d1f01b1a188de54b4c6458e1aa456696e566e015fcb570135517848c0256` |
| prompt | `e74bb5e1253af107ff27badfcfaca747b94574e19677d22cfe42b0b1c0ba5633` |
| tool schema | `f33f61d48bc4341f88077e90afca941819769b6209eb54893a9ed6b44856aba5` |

正式比较以本表及 artifact receipt 中的完整 hash 为准。

## 5. 后续实验最低报告要求

任何进入主表的新 checkpoint，至少同时报告：

1. run/config/code/checkpoint/Qwen/RP adapter/prompt/tool schema 的完整身份；
2. Adapter 是 frozen 还是 trainable；full Qwen 的 vision encoder、merger 与 LM 是否更新；
3. 数据 manifest、BS、rollout n、world size、micro-batch、GA、LR 和 reward 分解；
4. 本文九行分项表与 canonical Macro*；
5. normal final、direct answer、call cap、invalid format、max/context tokens 的数量和比例；
6. 平均/中位 response tokens、tool calls、成功 observation 数；
7. judge parse failure、judge/API/system failure 分开报告；
8. 同协议 Step 0 与目标 checkpoint；使用的 paired RNG namespace 或其尚未启用的明确声明；
9. legacy-RNG 结果若小于当前 `1.01 pp` 单次波动参照，只能作为趋势；paired 结果也必须报告 seed 数和每题采样数。

## 6. RP67 T-free Step 8 → Step 16 实验身份（已完成）

本组实验的目的是判断 Step 8 的变化是短暂峰值、随机波动，还是可继续的 RL 信号。执行已完成，实际结果见第 7 节；以前的时间估算、旧 launcher 限制和待实现描述已删除，不再作为当前状态。

实验保持了以下科学身份：

- 从 Step 8 的完整 model、optimizer、scheduler、data cursor 和 RNG state 原位恢复；
- RP67 Adapter 保持 frozen，full Qwen（包括视觉路径）继续更新；
- reward 保持 T-free：answer correctness + protocol/tool-error penalty + repeated-call penalty；tool utility、focus、grounding 关闭；
- 保持 BS16 prompts × n16、world8、LR `1e-6`、constant scheduler；
- 永久保留 Step 8 与 Step 16；
- Step 0/8/16 使用同一 `temperature=1` paired seed namespace 重新评测。

## 7. RP67 T-free Step 0 / Step 8 / Step 16 paired-seed-v1 结果

### 7.1 结果身份

本次一次性评测了三个 checkpoint：Step 0、Step 8 和 Step 16。三臂继续使用第 1、2 节冻结的 CoreDev-2511 协议：同一 2,511-row manifest（实际推理 2,240 条单图，显式 hold 271 条多图）、同一 prompt、TGVF tool schema、`temperature=1` sampling、VLMEvalKit scorer 和七分量 Macro* 聚合。

本次唯一有意改变的是随机流身份。三臂共同使用：

```text
mode = common_random_numbers_per_task_turn
master_seed = 42
seed_namespace = coredev2511-official-v1/rp67-tfree/step0-step8-step16/temp1/seed42/v1
protocol_sha256 = e82f05a663928df20e5a757c2de14264c990cc04cb9bf4985e23f1e90e257a25
```

seed 由 task/sample/rollout/assistant-turn 身份导出，并明确排除 evaluation ID、arm 名、optimizer step、checkpoint hash、policy weight hash 与 prompt-token hash。2,240 条共同推理样本的三臂 paired stream identity mismatch 为 `0`。

旧结果与新结果使用的是相同模型权重，不是换了 checkpoint：

- Step 0 的 combined/Qwen/RP67 身份哈希在两次评测中直接一致；
- Step 8 的旧、新导出采用不同 shard layout，因此文件/tree hash 不同；逐 named-tensor 核验覆盖 `750/750` tensors、`8,767,123,696` 个 bf16 elements，key/shape/dtype mismatch 为 `0`，`torch.equal` mismatch 为 `0`；
- RP67 state 均为 `f223d1f01b1a188de54b4c6458e1aa456696e566e015fcb570135517848c0256`。

因此，同一 Step 在 legacy 与 paired 块之间出现的分数差异不能归因于模型权重变化；主要区别是 `temperature=1` 的采样随机流。

### 7.2 三 checkpoint 配对结果

所有数字单位为 `%`；Macro* 使用未四舍五入的分量计算。

| benchmark | paired Step 0 | paired Step 8 | paired Step 16 |
|---|---:|---:|---:|
| VStarBench Overall | 62.83 | 65.45 | 64.92 |
| HRBench Average / all | 58.50 | 60.00 | 64.50 |
| BLINK single-image（180） | 62.78（113/180） | 60.56（109/180） | 63.33（114/180） |
| OCRBench v2 English | 46.20 | 44.33 | 43.83 |
| OCRBench v2 Chinese | 40.87 | 34.45 | 37.95 |
| OCR EN/CN mean | 43.53 | 39.39 | 40.89 |
| MMMU-Pro single-image（269） | 50.19（135/269） | 47.58（128/269） | 47.96（129/269） |
| MathVista MINI | 68.00 | 68.00 | 70.00 |
| MathVerse five-version macro | 53.40 | 52.40 | 55.80 |
| **Macro\*** | **57.0320** | **56.1964** | **58.1996** |

有效的同块 delta 为：

| 对比 | Macro* delta | 解释 |
|---|---:|---|
| Step 8 − Step 0 | -0.84 pp | paired 单次负向变化 |
| Step 16 − Step 8 | +2.00 pp | 继续训练后明显回升 |
| Step 16 − Step 0 | +1.17 pp | 当前支持 RL 有效的正向信号 |

### 7.3 与 legacy-RNG 的边界

| RNG block | Step 0 | Step 8 | Step 16 |
|---|---:|---:|---:|
| `legacy-RNG` | 56.37 | 57.28 | — |
| `paired-seed-v1` | 57.0320 | 56.1964 | 58.1996 |

合法的主结论必须来自完整 paired 块 `57.0320 → 56.1964 → 58.1996`。禁止用 legacy Step 8 `57.28` 与 paired Step 16 `58.1996` 相减，也禁止在不写 RNG block 的情况下只报告“RP67 Step 0/8”。

paired common-random-numbers 显著改善了 checkpoint 间 delta 的可比性，但没有把 `temperature=1` 变成确定性评测：每题仍只有一次采样，且不同 checkpoint 的 token 分布会使轨迹逐步分叉。因此 `Step 16 − Step 0 = +1.17 pp` 当前应表述为“正向信号，支持继续验证 RL 有效”，不能表述为已经统计确认的稳定增益。Step 0/8 的 legacy 与 paired 分数分别相差 `+0.66 pp` 和 `-1.08 pp`，也直接说明 temp=1 单次评测仍存在足以影响约 1 pp 结论的波动。

## 8. PRL18-R0 joint/unfrozen RP67 T-free Step 8 / Step 16 paired 结果

PRL18-R0 在 full-Qwen T-free 训练中同时更新 RP67 Adapter。canonical 结果来自：

```text
artifacts/policy/PRL-18-R0-qwen3-instruct-full-joint-rp67-bs16-n16-tfree-novisual-8step-ws8/
  evaluation/PRL18-R0-JOINT-RP67-TFREE-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1/
    paired-summary.json
```

本评测复用第 7 节的 common-random-numbers 协议：`master_seed=42`、`mode=common_random_numbers_per_task_turn`、seed namespace `coredev2511-official-v1/rp67-tfree/step0-step8-step16/temp1/seed42/v1`，protocol SHA256 为 `e82f05a663928df20e5a757c2de14264c990cc04cb9bf4985e23f1e90e257a25`。因此 joint Step 8/16、共同 Step 0 与 frozen Step 8/16 之间是同协议的有效 paired 比较。两臂评测均为 `pass`：每臂实际推理 2,240 条单图，显式 hold 271 条多图，judge parse failure 均为 `0`。

所有数字单位为 `%`；Macro* 使用未四舍五入的七分量计算。

| benchmark | joint Step 8 | joint Step 16 | Step 16 − Step 8 |
|---|---:|---:|---:|
| VStarBench Overall | 64.3979058 | 57.5916230 | -6.8062828 |
| HRBench Average / all | 61.5000000 | 67.0000000 | +5.5000000 |
| BLINK single-image（180） | 64.4444444（116/180） | 62.2222222（112/180） | -2.2222222 |
| OCRBench v2 English | 42.8649092 | 42.2652333 | -0.5996759 |
| OCRBench v2 Chinese | 29.8650025 | 30.2947793 | +0.4297768 |
| OCR EN/CN mean | 36.3649558 | 36.2800063 | -0.0849495 |
| MMMU-Pro single-image（269） | 47.5836431（128/269） | 49.0706320（132/269） | +1.4869889 |
| MathVista MINI | 67.3333333 | 68.3333333 | +1.0000000 |
| MathVerse five-version macro | 51.8000000 | 55.2000000 | +3.4000000 |
| **Macro\*** | **56.2034689** | **56.5282595** | **+0.3247906** |

同一 paired block 下的关键对照为：

| 对比 | Macro* delta |
|---|---:|
| joint Step 8 − common Step 0（57.0320） | -0.8285769 pp |
| joint Step 16 − common Step 0（57.0320） | -0.5037863 pp |
| joint Step 8 − frozen Step 8（56.1964） | +0.0070686 pp |
| joint Step 16 − frozen Step 16（58.1996） | -1.6713050 pp |

结论：当前结果不支持 joint Adapter update。joint Step 16 虽较自身 Step 8 增加 `+0.3248 pp`，但仍低于共同 Step 0，并显著落后 frozen Step 16；frozen Step 16 仍是最佳 checkpoint。分项表现是能力重分配而非一致提升，最明显的是 VStar `-6.81 pp`，同时 HR `+5.50 pp`、MathVerse `+3.40 pp`、MMMU-Pro `+1.49 pp`。`temperature=1` 单次采样的统计边界仍适用。
