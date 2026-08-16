# Policy RL CoreDev-2511 统一测量标准与主基线

日期：2026-08-12

结果更新：2026-08-16

状态：`PRIMARY MEASUREMENT CONTRACT / FROZEN V1`

Contract ID：`POLICY-RL-COREDEV2511-MEASUREMENT-20260812-v1`

适用范围：后续 Qwen3-VL-8B-Instruct Crop / TGVF policy-RL 的 Step 0、训练中间点与最终 checkpoint 对比。

本文冻结两件事：

1. CoreDev-2511 的统一测量与聚合口径；
2. canonical 大表的汇报结构；实验结果持续按同一契约追加，当前更新至 2026-08-16。

本文只替代旧文档中的 headline 聚合值，不否定旧文档记录的模型、prompt、checkpoint、训练配置与 artifact 身份。特别是 `docs/POLICY_RL_PRIMARY_BASELINE_20260810.md` 中使用 HRBench cycle 0 和 OCR Chinese-only 得到的旧均值，不再作为主汇报值。

PRL13--PRL20 small-batch 阶段的统一结论、证据强度与后续边界，见
[`POLICY_RL_SMALL_BATCH_PILOT_CLOSEOUT_20260814.md`](POLICY_RL_SMALL_BATCH_PILOT_CLOSEOUT_20260814.md)。

Crop Step 0/8/16 与 PRL21 Crop T-free Step 8/16 的同标准 `legacy-RNG` 结果位于第 3.1 节；
纯 TGVF 的 Frozen、Joint、Focus/Grounding 与 Teacher25 完整 paired 总表位于第 3.2 节；
Atomic Crop+TGVF 的 no-Teacher / Teacher25 完整 paired 总表位于第 3.3 节。各实验的详细
解释和 artifact 身份位于第 7--11 节。这些结果不修改本文件冻结的 benchmark、scorer、
prompt、sampling 或聚合契约；legacy-RNG、不同 paired namespace 与不同 prompt/tool
schema 必须按身份分别引用。

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
policy data = Teacher25 for the next mainline pilots
              75% retained T1 + 25% retained Stage1 teacher data
              no-Teacher remains the required scientific control
```

八条主要轨迹的一页摘要如下。Delta 只在同一行、同一 RNG/protocol block 内计算；
Crop、纯 TGVF 与 Crop+TGVF 之间不做严格 paired delta。

| 线路 / RNG block | Step 0 | Step 8 | Step 16 | 同块主要 delta |
|---|---:|---:|---:|---:|
| Crop clean → clean-final / legacy | 55.5742 | **59.7161** | 59.5502 | S16−S0 **+3.9760 pp** |
| Crop T-free PRL21 / legacy† | 55.5742‡ | **61.1032** | 61.0862 | S16−S8 -0.0170 pp；平台 |
| RP67 T-free Frozen / paired | 57.0320 | 56.1964 | **58.1996** | S16−S0 **+1.1675 pp** |
| RP67 T-free Frozen + Teacher25 / paired | 57.0320†† | 58.4655 | **59.9590** | S16−S0 **+2.9270 pp**；vs no-Teacher S16 **+1.7595 pp** |
| RP67 T-free Joint / paired | 57.0320 | 56.2035 | 56.5283 | S16−S0 -0.5038 pp |
| RP67 Frozen + F/G / paired | 57.0320 | **57.8849** | 57.5422 | S16−S0 +0.5102 pp |
| RP67 Frozen T-free Crop+TGVF / own paired | — | **62.1168** | 60.9539 | S16−S8 -1.1629 pp |
| RP67 Frozen T-free Crop+TGVF + Teacher25 / own paired | — | 62.3719 | **62.4974** | S16−S8 +0.1255 pp；vs no-Teacher S16 **+1.5434 pp** |

† PRL21 的实际 checkpoint path 与 weights SHA256 已核验正确，但旧 full-model evaluator 把 PRL13 protocol 身份同时写成了 checkpoint owner；因此它是 `pass_with_legacy_owner_binding`，不能称为 clean owner-bound canonical artifact。

‡ PRL21 没有重新推理 Step 0；这里仅复用协议相同的历史 Crop clean Step 0 作为 pilot 上下文，不是 paired 起点。

†† PRL22-A 没有重新推理 Step 0；数据混合不改变初始化，因此引用与 PRL17-R2 完全相同的
common paired Step 0。PRL22-B 没有组合工具 Step 0，不得补用该值。

| 决策 | 当前证据 | 证据边界 |
|---|---|---|
| **Frozen Adapter** | 当前 single-seed matched control 下，Frozen S16 `58.1996`，Joint S16 `56.5283`，Frozen 高 **`1.6713 pp`**；Joint S16 还低于 common S0 `57.0320` | 这是目前最强、最直接的 control evidence；支持当前 RL 默认冻结 RP67、只更新 full Qwen policy，但不是多 seed 统计确认 |
| **T-free reward** | legacy 筛选中，`+T` 为 `57.38 → 56.30`，T-free 为 `56.37 → 57.28`；随后 T-free paired S16 相对 common S0 为 **`+1.1675 pp`** | `+T` 与 T-free 尚不是共享随机流的严格 reward ablation；因此 T-free 是当前最受支持的默认，不写成已经统计证明的独立因果结论 |
| **RP67** | RP67 是当前进入 paired policy-RL 主线并取得最佳 TGVF checkpoint 的 Stage 1 版本；其 image-axis 目标提供结构动机 | 本文没有提供 RP66 vs RP67 同 `paired-seed` 严格结果；因此 RP67 是当前工程默认，不是本表已单独证明优于 RP66 的结论 |
| **Crop+TGVF** | Atomic 组合在自己的 paired block 中完成 S8/S16，S8 Macro* `62.1168`，证明组合协议兼容且具有很强 pilot 表现 | 没有组合 S0；Crop、纯 TGVF 与组合工具的 prompt/tool/RNG block 不同，不能把跨线路差值称为严格 synergy |
| **Teacher25 policy data** | 纯 TGVF 同步提升 S8 `+2.2691 pp`、S16 `+1.7595 pp`；Crop+TGVF 提升 S8 `+0.2550 pp`、S16 `+1.5434 pp`，并把 S8→S16 从 `-1.1629` 改为 `+0.1255 pp` | 两条工具线、四个 endpoint 均正向，是 strongly positive pilot evidence；仍只有一个 training seed、每题一次 temp1 采样，且 historical controls 与 treatment 不是 byte-identical executable commit |

因此，本文后续出现“当前 TGVF 默认线”时，均指 **RP67 + T-free + Frozen Adapter**；
后续主线 pilot 默认优先采用 **Teacher25**，同时保留 no-Teacher 作为显式 scientific
control。解冻 Adapter 不再作为默认设定；若再测，必须作为显式 ablation。

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

## 3. Canonical 大表（更新至 2026-08-16）

### 3.1 Legacy-RNG 历史总表

所有数字单位为 `%`。OCR mean 是 EN/CN 的均值；Macro* 只把 OCR mean 计入一次。

注意：本节所有工具 arm 都保留自历史 `legacy-RNG` 评测，seed 会随 evaluation ID 改变。它们共享本文的 benchmark/scorer/Macro* 测量标准，但不共享 paired 随机流。它们不能被无标签地替换为第 7 节的 `paired-seed-v1` 数值，也不能与 paired 结果跨块计算 delta。Original 还使用 direct prompt，只是端到端参考。PRL21 两列另带 `legacy owner-binding` 限制，实际权重正确，但 owner provenance 不完整。

| benchmark | Original | Crop clean S0 | Crop clean-final S8 | Crop clean-final S16 | Crop T-free S8† | Crop T-free S16† | RP67 +T S0 | RP67 +T S8 | RP67 T-free S0 | RP67 T-free S8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VStarBench Overall | 50.79 | 78.01 | 76.96 | 79.06 | 75.39 | 75.92 | 66.49 | 58.64 | 64.92 | 65.45 |
| HRBench Average / all | 59.00 | 53.50 | 62.50 | 60.00 | 71.00 | 73.50 | 59.50 | 60.00 | 58.00 | 62.50 |
| BLINK single-image（180） | 65.56 | 57.22 | 60.00 | 57.78 | 62.78 | 62.78 | 59.44 | 63.89 | 63.33 | 64.44 |
| OCRBench v2 English | 49.89 | 40.46 | 47.39 | 46.80 | 49.17 | 49.79 | 46.12 | 44.99 | 45.47 | 44.54 |
| OCRBench v2 Chinese | 46.48 | 37.45 | 51.21 | 51.11 | 53.59 | 57.66 | 34.19 | 37.66 | 36.35 | 37.83 |
| OCR EN/CN mean | 48.19 | 38.95 | 49.30 | 48.95 | 51.38 | 53.72 | 40.16 | 41.33 | 40.91 | 41.19 |
| MMMU-Pro single-image（269） | 39.03 | 43.87 | 47.58 | 50.93 | 46.84 | 45.35 | 48.33 | 47.58 | 45.35 | 48.33 |
| MathVista MINI | 74.33 | 62.67 | 67.67 | 65.33 | 65.33 | 65.33 | 73.33 | 65.67 | 68.67 | 65.67 |
| MathVerse five-version macro | 50.60 | 54.80 | 54.00 | 54.80 | 55.00 | 51.00 | 54.40 | 57.00 | 53.40 | 53.40 |
| **Macro\*** | **55.36** | **55.5742** | **59.7161** | **59.5502** | **61.1032** | **61.0862** | **57.38** | **56.30** | **56.37** | **57.28** |

† PRL21 Crop T-free 是 `temperature=1 / legacy-RNG / pass_with_legacy_owner_binding`。该标记限制 provenance 声明，不表示评分失败。

对应的 RL delta：

| 线路 | Step 0 | Step 8 | Step 16 | 同 legacy 块的观测变化 | 当前解释 |
|---|---:|---:|---:|---:|---|
| Crop clean-final | 55.5742 | 59.7161 | 59.5502 | S8−S0 **`+4.1419 pp`**；S16−S0 **`+3.9760 pp`**；S16−S8 `-0.1659 pp` | 明显正向 pilot；S8/S16 是平台期，不应将 `-0.17 pp` 解读为真实退化 |
| Crop T-free PRL21† | 55.5742‡ | 61.1032 | 61.0862 | S8−S0 **`+5.5290 pp`**；S16−S0 **`+5.5120 pp`**；S16−S8 `-0.0170 pp` | 明显正向 pilot；S8/S16 完全平台，但历史 S0 复用与 legacy-RNG 不允许严格因果解释 |
| RP67 +T | 57.38 | 56.30 | — | S8−S0 `-1.08 pp` | 单次负向趋势；幅度接近旧 RNG 波动参照，不能单独定性 |
| RP67 T-free | 56.37 | 57.28 | — | S8−S0 `+0.91 pp` | 单次正向趋势；本块单独不足以证明 reward 优势，需结合第 3.2 节 paired 结果 |

Crop S16 的 BLINK 和 MMMU-Pro 分别为 `104/180` 和 `137/269`。Crop 的 S0/S8/S16 共享同一 CoreDev manifest、Crop prompt/runtime、sampling 参数与 scorer 契约，但 evaluation ID 不同，因此仍是 legacy 同标准比较，不是 common-random-numbers 严格 paired 对照。

‡ PRL21 未重新推理 Step 0；表中 `55.5742` 只复用协议相同的历史 Crop clean Step 0 作为 pilot 上下文。

#### 3.1.1 Superseded RP66 formal records

旧 RP66 formal evaluations 也属于项目的有效历史证据，不能因为主线已切换到 RP67 而
从 canonical ledger 消失。下表按本文冻结后的 HR Average/all、OCR EN/CN mean、
BLINK-180、MMMU-269 与 MathVerse five-version 口径重新聚合；它取代旧 primary report
中 HR cycle-0 / OCR Chinese-only 的过时 headline。

| benchmark | PRL15 joint S0 | PRL15 joint S8 | PRL16-F2 frozen S0 | PRL16-F2 frozen S8 | PRL17-R0 frozen S4 | PRL17-R0 frozen S8 |
|---|---:|---:|---:|---:|---:|---:|
| VStarBench Overall | 65.4450 | 60.2094 | 60.2094 | 61.7801 | 62.3037 | 62.3037 |
| HRBench Average / all | 62.5000 | 58.0000 | 58.0000 | 58.5000 | 55.0000 | 57.5000 |
| BLINK single-image（180） | 62.2222 | 56.1111 | 63.8889 | 58.8889 | 55.0000 | 61.6667 |
| OCRBench v2 English | 48.7189 | 44.4560 | 43.5916 | 45.4730 | 43.2998 | 44.8352 |
| OCRBench v2 Chinese | 35.8985 | 31.3578 | 32.4580 | 36.8703 | 32.9896 | 35.2487 |
| OCR EN/CN mean | 42.3087 | 37.9069 | 38.0248 | 41.1717 | 38.1447 | 40.0420 |
| MMMU-Pro single-image（269） | 42.0074 | 45.3532 | 43.1227 | 44.2379 | 42.7509 | 41.6357 |
| MathVista MINI | 67.0000 | 64.6667 | 64.6667 | 62.3333 | 65.0000 | 63.0000 |
| MathVerse five-version macro | 48.4000 | 50.0000 | 49.6000 | 46.2000 | 48.6000 | 49.6000 |
| **Macro\*** | **55.6976** | **53.1782** | **53.9303** | **53.3017** | **52.3999** | **53.6783** |

| Historical RP66 run | Valid within-run delta | Interpretation |
|---|---:|---|
| PRL15-R1 joint/trainable RP66 | S8−S0 **-2.5194 pp** | 正式负向结果；旧 reward/runtime 与当前 RP67 主线不同 |
| PRL16-F2 frozen RP66 +T | S8−S0 **-0.6286 pp** | 小幅负向；legacy temp1 单次结果，不作统计确认 |
| PRL17-R0 frozen RP66 +T | S8−S4 **+1.2784 pp** | S4→S8 正向，但缺失同协议 S0，不能据此做 RP66-vs-RP67 因果比较 |

三组均为 `legacy-RNG`，彼此也不是共享随机流。PRL15 Step 0/8 official summaries 为
`pass`，judge parse failures 为 `3/4`；PRL16-F2 Step 0/8 为 `pass`，parse failures
为 `2/2`；PRL17-R0 Step 4/8 为 `pass`，parse failures 为 `5/5`。这些记录保留历史
结论，但不改变 RP67 Frozen + T-free 的当前默认身份。

PRL16-F1 的 Step 0/1/2 artifact 明确标记为 `DIAGNOSTIC`，因此不进入 accuracy headline：

```text
PRL16-F1-FROZEN-RP66-COREDEV2511-STEP0-STEP1-STEP2-DIAGNOSTIC-V1
paired-summary.json SHA256 = 13c66ffbcd91f48cc6803bc4d060f0849e57c98ce73ddfb0543ba02b06da5342
```

Canonical historical artifact identities：

```text
PRL15-R1:
artifacts/policy/PRL-15-R1-qwen3-instruct-full-rp66-bs16-n16-crop16-math-equiv-ws4/
  evaluation/PRL15-R1-RP66-COREDEV2511-STEP0-STEP8-SAME-PROTOCOL-RUNTIMEFIX-V2/
paired-summary.json SHA256 = b4621ca57a58eeb518d5ff0d46ac4e6979131f9d677472882ad92c9bea78b258

PRL16-F2:
artifacts/policy/PRL-16-F2-qwen3-instruct-full-frozen-rp66-bs16-n16-t1-crop16-lifecycle-fix-8step-ws8/
  evaluation/PRL16-F2-FROZEN-RP66-COREDEV2511-STEP0-STEP8-SAME-PROTOCOL-V1/
paired-summary.json SHA256 = ed5f5ee38d37cd65b3700605079e1edc88408e62045cdd4e976f01c48f41ecdc

PRL17-R0:
artifacts/policy/PRL-17-R0-qwen3-instruct-full-frozen-rp66-bs16-n16-t1-shaped-novisual-8step-ws8/
  evaluation/PRL17-R0-FROZEN-RP66-COREDEV2511-STEP4-STEP8-SAME-PROTOCOL-V1/
paired-summary.json SHA256 = 6ae44bd4e32a8967e1d5fb0575ed6b6cb2b5cf2504e5e201a42b050d5c1eabaa
evaluation-complete SHA256 = 559127b96bd7dfdb68740a46b5b5f3f5555e48c2d007ba32b2604f9130953013
```

### 3.2 Paired-seed-v1 纯 TGVF 完整总表

下表统一收录当前纯 TGVF paired 主线的全部正式 checkpoint：no-Teacher Frozen、
Joint/unfrozen、Focus/Grounding visual reward，以及 Teacher25 Frozen。它们使用相同
CoreDev-2511 manifest、推理协议、`temperature=1`、`master_seed=42` 与
common-random-numbers namespace。未重复推理 Step 0 的 arm 共享完全相同的初始化
`57.0320`；该共享只适用于纯 TGVF，不适用于 Crop+TGVF。

所有数字单位为 `%`；Macro* 使用未四舍五入的七个分量计算。OCR English/Chinese
分别展示，但二者的 mean 在 Macro* 中只占一个分量。

| benchmark | Common S0 | Frozen S8 | Frozen S16 | Joint S8 | Joint S16 | F/G S8 | F/G S16 | Teacher25 S8 | Teacher25 S16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VStarBench Overall | 62.83 | 65.45 | 64.92 | 64.40 | 57.59 | 68.06 | 66.49 | **71.20** | 69.11 |
| HRBench Average / all | 58.50 | 60.00 | 64.50 | 61.50 | 67.00 | 63.50 | 60.00 | 61.00 | **69.50** |
| BLINK single-image（180） | 62.78 | 60.56 | 63.33 | **64.44** | 62.22 | 63.33 | 60.56 | 61.11 | 60.56 |
| OCRBench v2 English | 46.20 | 44.33 | 43.83 | 42.86 | 42.27 | 45.20 | 45.55 | **46.14** | 45.83 |
| OCRBench v2 Chinese | 40.87 | 34.45 | 37.95 | 29.87 | 30.29 | 38.77 | **41.75** | 41.63 | 36.23 |
| OCR EN/CN mean | 43.53 | 39.39 | 40.89 | 36.36 | 36.28 | 41.98 | 43.65 | **43.89** | 41.03 |
| MMMU-Pro single-image（269） | **50.19** | 47.58 | 47.96 | 47.58 | 49.07 | 44.98 | 48.70 | 45.72 | **50.19** |
| MathVista MINI | 68.00 | 68.00 | 70.00 | 67.33 | 68.33 | 67.33 | 69.00 | 69.33 | **73.33** |
| MathVerse five-version macro | 53.40 | 52.40 | 55.80 | 51.80 | 55.20 | 56.00 | 54.40 | **57.00** | 56.00 |
| **Macro\*** | **57.0320** | **56.1964** | **58.1996** | **56.2035** | **56.5283** | **57.8849** | **57.5422** | **58.4655** | **59.9590** |

同一 pure-TGVF paired block 内的训练轨迹：

| Arm | Step 8 − S0 | Step 16 − S0 | Step 16 − Step 8 |
|---|---:|---:|---:|
| Frozen no-Teacher | -0.8356 pp | +1.1675 pp | +2.0032 pp |
| Joint / unfrozen | -0.8286 pp | -0.5038 pp | +0.3248 pp |
| Frozen + Focus/Grounding | +0.8529 pp | +0.5102 pp | -0.3427 pp |
| Frozen + Teacher25 | **+1.4335 pp** | **+2.9270 pp** | **+1.4935 pp** |

同 endpoint 的 matched treatment 对照：

| Treatment − Frozen no-Teacher | Step 8 | Step 16 |
|---|---:|---:|
| Joint / unfrozen | +0.0071 pp | -1.6713 pp |
| Focus/Grounding | +1.6885 pp | -0.6573 pp |
| Teacher25 | **+2.2691 pp** | **+1.7595 pp** |

因此当前 paired 证据同时支持两项主线决策：冻结 RP67 Adapter，以及把 Teacher25
作为下一阶段默认数据候选。F/G 的 accuracy 结果不单调，仍需独立 visual-quality
audit，不能仅凭 Step 8 的增益宣布有效。

### 3.3 Paired-seed-v1 Atomic Crop+TGVF 完整总表

PRL20 no-Teacher 与 PRL22-B Teacher25 使用 Atomic Crop+TGVF 自己的相同 paired
测量协议。该 block 没有组合工具 Step 0；不得从纯 TGVF 或 Crop 线路补造起点。

| benchmark | No Teacher S8 | Teacher25 S8 | Delta S8 | No Teacher S16 | Teacher25 S16 | Delta S16 |
|---|---:|---:|---:|---:|---:|---:|
| VStarBench Overall | 70.16 | **71.20** | +1.05 | 71.73 | **72.77** | +1.05 |
| HRBench Average / all | **73.00** | 69.00 | -4.00 | 68.00 | **70.00** | +2.00 |
| BLINK single-image（180） | 65.00 | **67.22** | +2.22 | 61.67 | **68.33** | +6.67 |
| OCRBench v2 English | 52.47 | **52.93** | +0.45 | 51.53 | **53.05** | +1.52 |
| OCRBench v2 Chinese | **54.41** | 52.21 | -2.19 | **53.41** | 51.84 | -1.57 |
| OCR EN/CN mean | **53.44** | 52.57 | -0.87 | **52.47** | 52.45 | -0.02 |
| MMMU-Pro single-image（269） | 47.96 | **51.67** | +3.72 | **49.81** | 45.72 | -4.09 |
| MathVista MINI | **69.67** | 69.33 | -0.33 | 67.00 | **71.00** | +4.00 |
| MathVerse five-version macro | 55.60 | 55.60 | 0.00 | 56.00 | **57.20** | +1.20 |
| **Macro\*** | **62.1168** | **62.3719** | **+0.2550** | **60.9539** | **62.4974** | **+1.5434** |

| Crop+TGVF trajectory | Step 16 − Step 8 |
|---|---:|
| No Teacher | -1.1629 pp |
| Teacher25 | **+0.1255 pp** |

Step 8 的 treatment gap 很小，单独不足以定性；Step 16 的 `+1.5434 pp` 更有意义，
并且 Teacher25 把 no-Teacher 的后半程退化改为基本持平。结合第 3.2 节，Teacher25
在两条工具协议、四个 endpoint 的 Macro* 全部正向。

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

### Crop T-free PRL21 Step 8 / Step 16

```text
artifacts/policy/
  PRL-21-R0-qwen3-instruct-full-crop-bs16-n16-tfree-16step-ws8/
    evaluation/
      PRL21-R0-CROP-TFREE-COREDEV2511-STEP8-STEP16-TEMP1-SEED42-V1/
        evaluation-summary.json
        step8/scoring/coredev-official-v1-recovery2/
          coredev-2511-eval-summary.json
        step16/scoring/coredev-official-v1-recovery2/
          coredev-2511-eval-summary.json
```

该 `evaluation-summary.json` 的状态为 `pass_with_legacy_owner_binding`。实际 checkpoint weights SHA256 为：

| checkpoint | policy weights SHA256 |
|---|---|
| PRL21 Crop T-free Step 8 | `a74e460e2a4ace4a2e7dfbc22530e4dc46a542df399c2e8e659fe764680d66cc` |
| PRL21 Crop T-free Step 16 | `7f5012a9e4346adda801d4a0fd94a9b3cfc05e05c30a18095e51210c2c7407db` |

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

### Current paired artifact index

| Experiment | Canonical evaluation directory | `paired-summary.json` SHA256 | `evaluation-complete` SHA256 |
|---|---|---|---|
| PRL17-R2 Frozen no-Teacher TGVF | `PRL17-R2-FROZEN-RP67-TFREE-COREDEV2511-STEP0-STEP8-STEP16-PAIRED-SEED-V1` | `bf90a99f52f1943509fa83b8c377c959d32699e5127021ea1b09c49941119176` | — |
| PRL18-R0 Joint TGVF | `PRL18-R0-JOINT-RP67-TFREE-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1` | `2c7172f60b343f5ecf0749ade451b4617dc1c415b041a309b45b2248a81fdfaa` | — |
| PRL19-R0 Focus/Grounding TGVF | `PRL19-R0-FROZEN-RP67-TFREE-VISUAL-API-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1` | `581c37c8e68d1dbe30e7be715e4dfd53fa8cf983b2b266b379fc1db16aaee156` | — |
| PRL20-R0 Atomic Crop+TGVF | `PRL20-R0-FROZEN-RP67-TFREE-CROP-TGVF-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1` | `e45fa6876facc994b212ed6d7b63eaae221732b4d6a0ede6cb93add88e56fe57` | `c92cf45fd5badf9de662848adeef2718b0e88fb8a35357ba2c3998a186b9053f` |
| PRL22-A Teacher25 TGVF | `PRL22-A-FROZEN-RP67-TFREE-TEACHER25-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1` | `06b7ccbc4c2d71d9bb338057a62da4b95e8bdf36b35f59eb5de9d3e101efd03e` | `c4c49a7c21c970b563f221c863c549cb62b143971633a1db8014cdb184af2b20` |
| PRL22-B Teacher25 Crop+TGVF | `PRL22-B-R0-FROZEN-RP67-TFREE-CROP-TGVF-TEACHER25-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1` | `dc07a11bbad97044294f634a188e5038b44a079fa964e34fc2cd63a9c61a5ee5` | `4ea1f27251856b588f4ae2f8d52c6e1de9116759adbae0b1b013c52f8de871aa` |

详细解释与实例索引：

- [`POLICY_RL_SMALL_BATCH_PILOT_CLOSEOUT_20260814.md`](POLICY_RL_SMALL_BATCH_PILOT_CLOSEOUT_20260814.md)
- [`PRL21_CROP_TFREE_16STEP_RESULTS_AND_EVALUATION_INCIDENT_20260815.md`](PRL21_CROP_TFREE_16STEP_RESULTS_AND_EVALUATION_INCIDENT_20260815.md)
- [`PRL22_TEACHER25_POLICY_DATA_ABLATION_RESULTS_20260816.md`](PRL22_TEACHER25_POLICY_DATA_ABLATION_RESULTS_20260816.md)
- [`POLICY_RL_MAINLINES_ACTUAL_INFERENCE_EXAMPLES_20260814.md`](POLICY_RL_MAINLINES_ACTUAL_INFERENCE_EXAMPLES_20260814.md)

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
3. policy-data parent/schedule manifest 与 hashes、source mixture、sampling seed、是否 replacement，以及 BS、rollout n、world size、micro-batch、GA、LR 和 reward 分解；
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

结论：当前结果不支持 joint Adapter update。joint Step 16 虽较自身 Step 8 增加
`+0.3248 pp`，但仍低于共同 Step 0，并显著落后 frozen Step 16；在这个纯 TGVF
Frozen/Joint control block 中，frozen Step 16 仍是最佳 checkpoint。分项表现是能力
重分配而非一致提升，最明显的是 VStar `-6.81 pp`，同时 HR `+5.50 pp`、
MathVerse `+3.40 pp`、MMMU-Pro `+1.49 pp`。`temperature=1` 单次采样的统计边界
仍适用。

## 9. PRL19 Focus/Target + Grounding visual reward

PRL19 保持 RP67、Frozen、T-free、BS16 × n16、world8、LR `1e-6` 与 policy/tool
协议不变，只开启 gold-free Focus/Target 和 Grounding reward。其 matched paired
Macro* 为：

| checkpoint | Common S0 | No-visual | Visual | Visual − No-visual |
|---|---:|---:|---:|---:|
| Step 8 | 57.0320 | 56.1964 | **57.8849** | **+1.6885 pp** |
| Step 16 | 57.0320 | **58.1996** | 57.5422 | `−0.6573 pp` |

该结果不支持从 accuracy 单独判定 visual reward：Step 8 正向、Step 16 反向，且
accuracy 不能直接测量 target 是否更依赖图像或 hallucination 是否降低。默认配方继续
关闭 F/G；它们保留为需要独立 held-out foveation/grounding audit 的研究方向。

Canonical artifact：

```text
artifacts/policy/
  PRL-19-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-visual-api-8step-ws8/
    evaluation/
      PRL19-R0-FROZEN-RP67-TFREE-VISUAL-API-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1/
        paired-summary.json
```

summary SHA256：`581c37c8e68d1dbe30e7be715e4dfd53fa8cf983b2b266b379fc1db16aaee156`。

## 10. PRL20 Atomic Crop+TGVF paired 结果

PRL20 使用 `tgvf_crop_tool(bbox_2d,target)`：从不可变原图取得 crop，经 Qwen vision
与 frozen RP67 生成 `D`，policy 只收到 `D`，不收到 crop RGB。T-free reward、
BS16 × n16、world8 和 full-Qwen update 保持不变。Step 8/16 在组合工具自己的
paired namespace 内完成 2,240 条单图推理；271 条多图显式 hold。

| benchmark | Combo Step 8 | Combo Step 16 |
|---|---:|---:|
| VStarBench Overall | 70.1571 | 71.7277 |
| HRBench Average / all | 73.0000 | 68.0000 |
| BLINK single-image（180） | 65.0000 | 61.6667 |
| OCR EN/CN mean | 53.4388 | 52.4690 |
| MMMU-Pro single-image（269） | 47.9554 | 49.8141 |
| MathVista MINI | 69.6667 | 67.0000 |
| MathVerse five-version macro | 55.6000 | 56.0000 |
| **Macro\*** | **62.1168** | **60.9539** |

组合内部的有效 paired delta 为 Step 16 − Step 8 `−1.1629 pp`；PRL20 Step 8 是
no-Teacher Crop+TGVF control 内的最佳 checkpoint。加入 Teacher25 后的当前最高同协议
结果是 PRL22-B Step 16 的 `62.4974`，完整对照见第 3.3、11 节。PRL20 没有组合工具
Step 0，且其 prompt/tool/RNG block 不同于纯 TGVF，Crop 历史值又属于 legacy-RNG。
因此可以据此确认 Crop+TGVF 兼容并具有很强 pilot 表现，但不能把跨线路差值称为严格
synergy。

Canonical artifact：

```text
artifacts/policy/
  PRL-20-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-8step-ws8/
    evaluation/
      PRL20-R0-FROZEN-RP67-TFREE-CROP-TGVF-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1/
        paired-summary.json
```

summary SHA256：`e45fa6876facc994b212ed6d7b63eaae221732b4d6a0ede6cb93add88e56fe57`。

## 11. PRL22 Teacher25 policy-data ablation

### 11.1 数据合同

PRL22 只替换 policy-RL prompt population，目标是测试与 RP67 Stage1 表示分布更一致的
teacher questions 是否能改善 small-batch RL。Teacher25 的不可变 schedule 为：

| Item | Value |
|---|---|
| Existing parent | 77,541 retained prompts；VStar、ArxivQA、ThinkLite |
| Teacher parent | 24,779 retained prompts；ChartQA、DocVQA、TextOCR、TextVQA、Visual Genome |
| Materialized schedule | 20,480 prompts；seed 42；no replacement |
| Existing / Teacher | 15,360 / 5,120，即 75% / 25% |
| Every BS16 | 12 existing + 4 Teacher |
| Every BS256 macro | 90 VStar + 58 ArxivQA + 44 ThinkLite + 64 Teacher |
| Ordered role pattern | `old, old, old, teacher` |

```text
artifact = artifacts/data/policy_rl/PRL22-TEACHER25-MIXED-SCHEDULE-v1
manifest_file_sha256 = 48244271b9537700b22bc6f6bbe3322caa1f17ddeccb7dfc43c02729288b5662
content_sha256       = b5be9adfbd5ca7228a4d303aa900d0aec5ef877ceb7a7c328a15386ed2e3eab4
samples_sha256       = 040cb0f48ba5821b435e408024a81b9c90265d76a1fd5df1ecf73185c7e8439c
iteration_sha256     = ab47dca43fc669066629b1d94f5f34ec2147f3b7fc000226c88cd23907709e0b
```

Teacher rows 保留 `data_source=teacher` 与原始 `source_dataset` provenance，不伪装为
legacy source。两个 parent 共有 587 个 exact image hashes，但没有 exact
image-plus-question task overlap。Teacher pool 是 RP67 的 Stage1-train in-distribution
数据，因此该实验检验的是表示—policy 数据对齐，而不是任意外部数据混合。

### 11.2 控制变量身份

| Arm | Direct control | Tool protocol | Intended treatment |
|---|---|---|---|
| PRL22-A | PRL17-R2 Frozen TGVF | pure TGVF | no-Teacher → Teacher25 |
| PRL22-B | PRL20-R0 Atomic Crop+TGVF | `tgvf_crop_tool(bbox_2d,target)` | no-Teacher → Teacher25 |

两组均保持 Qwen3-VL-8B-Instruct full-policy update、frozen RP67 Step-2000 Adapter、
T-free reward、BS16 × n16、world8/micro2/GA1、AdamW constant LR `1e-6`、Step 8/16
endpoints 与对应工具线的 paired CoreDev 协议。PRL17 control executable commit 为
`2c1039e`，PRL20 control 为 `eadae55`，Teacher shared implementation 为 `37b99e2`；
所以这是 matched-recipe / matched-evaluation data ablation，不是 byte-identical-code
causal proof。

### 11.3 Cross-tool 结果

完整逐 benchmark 数值以第 3.2、3.3 节为 canonical 主表。Macro* 摘要为：

| Tool line | No Teacher S8 | Teacher25 S8 | Delta S8 | No Teacher S16 | Teacher25 S16 | Delta S16 |
|---|---:|---:|---:|---:|---:|---:|
| Pure TGVF | 56.1964 | **58.4655** | **+2.2691** | 58.1996 | **59.9590** | **+1.7595** |
| Atomic Crop+TGVF | 62.1168 | **62.3719** | **+0.2550** | 60.9539 | **62.4974** | **+1.5434** |

PRL22-A 相对共享 pure-TGVF Step 0 `57.0320`，Step 8 / 16 分别为 `+1.4335 pp` 与
`+2.9270 pp`。PRL22-B 则把 Step 8→16 从 no-Teacher 的 `-1.1629 pp` 改为
`+0.1255 pp`。A 的 Step 8/16 各有六个七分量 Macro* component 正向；B 的 Step 8
差异很小，但 Step 16 有五个七分量 component 正向。

四个 treatment official summary 均为 `pass`：PRL22-A Step 8/16 judge parse failures
为 `0/2`，PRL22-B 为 `0/1`，均低于各 benchmark fail threshold。

### 11.4 当前结论与证据边界

Teacher25 在两个工具协议、两个训练 endpoint 的 Macro* 全部正向，是本阶段
**strongly positive pilot evidence**。纯 TGVF 显示稳定 same-step 增益；Crop+TGVF
主要显示后半程稳定化。因此 Teacher25 成为后续 TGVF / Crop+TGVF pilot 的默认数据
候选，no-Teacher 继续作为 scientific control。

以下边界仍然保留：

- 每个 condition 只有一个 policy-training seed；
- `temperature=1` 下每题只有一次采样，paired RNG 降低但不消除采样波动；
- historical control 与 treatment 不是 byte-identical executable commit；
- Teacher 数据刻意与 RP67 对齐，结果不能直接外推到任意 representation/model；
- 部分 benchmark 仍有回落，例如 pure-TGVF S16 BLINK、Crop+TGVF S16 MMMU 与两条线的
  OCR Chinese。

因此可以宣布“Teacher25 对当前任务非常积极，并应进入下一阶段默认候选”，但不能写成
“已经完成多 seed 统计证明”。最干净的确认实验是 second training seed 或
same-current-commit no-Teacher rerun；PRL22-C pure Crop 可检验该作用是否依赖 TGVF。

### 11.5 Canonical artifacts

```text
PRL22-A:
artifacts/policy/PRL-22-A-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher25-8step-ws8/
  evaluation/PRL22-A-FROZEN-RP67-TFREE-TEACHER25-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1/

paired-summary.json SHA256 = 06b7ccbc4c2d71d9bb338057a62da4b95e8bdf36b35f59eb5de9d3e101efd03e
evaluation-complete SHA256 = c4c49a7c21c970b563f221c863c549cb62b143971633a1db8014cdb184af2b20

PRL22-B:
artifacts/policy/PRL-22-B-R0-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-teacher25-8step-ws8/
  evaluation/PRL22-B-R0-FROZEN-RP67-TFREE-CROP-TGVF-TEACHER25-COREDEV2511-STEP8-STEP16-PAIRED-SEED-V1/

paired-summary.json SHA256 = dc07a11bbad97044294f634a188e5038b44a079fa964e34fc2cd63a9c61a5ee5
evaluation-complete SHA256 = 4ea1f27251856b588f4ae2f8d52c6e1de9116759adbae0b1b013c52f8de871aa
```

PRL22-B 的 evaluation 曾遇到 transient vLLM port collision，并暴露 resume validator
把 runtime selector 与 serialized snapshot backend 混淆的问题。修复 commit 为
`566a1fe`；评测复用既有 Step-8/16 materialization，没有重训或重建 checkpoint。

更完整的数据、配置、逐项 delta 与 incident 记录见
[`PRL22_TEACHER25_POLICY_DATA_ABLATION_RESULTS_20260816.md`](PRL22_TEACHER25_POLICY_DATA_ABLATION_RESULTS_20260816.md)。
