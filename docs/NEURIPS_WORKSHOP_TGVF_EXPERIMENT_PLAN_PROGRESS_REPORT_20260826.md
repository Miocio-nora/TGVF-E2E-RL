# NeurIPS Workshop：TGVF 文章实验计划、推进台账与阶段报告

更新时间：2026-08-26（Asia/Tokyo）

状态：**实验进行中；RP67 三臂验证已闭合；TGVF/Atomic full-prompt 均已完成
2,240-row 生成与完整性核验；TGVF 与 Atomic full-prompt 七套官方评分均已闭合；
Atomic 双臂盲审包已就绪但人工标注尚未闭合。**

进度查看：本报告同步到 main 工作区
`docs/NEURIPS_WORKSHOP_TGVF_EXPERIMENT_PLAN_PROGRESS_REPORT_20260826.md`。在推理完成、评分完成、
审计包生成和文章结论更新等关键节点同步；运行中的计数只作为状态快照，不提前当作结果。按当前
授权，每个关键节点只提交这一个报告文件并 push 到 `origin/main`，不带入 main 的其他工作区改动。

## 1. 文章当前主线

本文不以宽泛的“互补能力与优化动态”作为唯一叙事。当前更可检验、也更有证据支撑的主线是：

> Target-conditioned latent evidence improves particular visual-reasoning regimes,
> especially semantic localization, relative depth, cross-region reasoning and
> visually grounded arithmetic, while retaining clear limitations on fine-grained
> text and pixel-faithful recognition.

正文以 **Pure TGVF** 为机制主线，以 **Native Crop** 为强工具基线，以 **Original** 为
raw direct 端到端参考。**Atomic Crop+TGVF** 固定列为探索性扩展：full-prompt 下总体仍高于
Original，但预冻结的 HR cross / Relative Reflectance 两个特异性优势未保留，已不满足正文核心
方法门槛；无偏 target 合格率审计继续用于界定探索性结果的语义质量，而不是事后恢复核心地位。

## 2. 固定术语和比较口径

| 简称 | 本文固定含义 | checkpoint / run | 解释边界 |
|---|---|---|---|
| **Original** | 原始 Qwen3-VL-8B-Instruct；无视觉工具、无自定义 system prompt | `PRL-04-R2-raw-instruct-coredev2511-gpu4567-r4` | 必须进入所有主表和 sub-benchmark 表；因 prompt/agent protocol 不同，只是端到端 direct reference，不是严格 paired control |
| **Crop** | PRL25-B native RGB Crop | S80，seed42 | 80-step 终点工具基线；不补 seed43 |
| **TGVF** | PRL25-C Pure TGVF，Frozen RP67 | S64，seed42；seed43 仅作所选 checkpoint 复测 | 文章机制主线 |
| **Atomic** | PRL25-D Atomic Crop+TGVF，Frozen RP67 | S16，seed42；seed43 仅作所选 checkpoint 复测 | 探索性扩展，不能在审计前声称已稳定学会高质量 target |
| **matched prompt** | 80-step 训练与既有 CoreDev 评测使用的简化、训练匹配 prompt | 历史结果 | 用于现有主表 |
| **full prompt** | 详细说明 target、bbox、关系与禁止答案泄漏的 Instruct prompt；可见与运行时上限均为 6 次 | `full_visual_tool_prompt_v5_instruct_cap6` | 只在冻结 S64/S16 上评测，不重选 checkpoint；衡量 prompt shift robustness |
| **Macro\*** | 七个百分比组件的无权平均 | VStar、HRBench、BLINK-180、OCR EN/CN mean、MMMU-269、MathVista、MathVerse 五版本宏平均 | 只在相同测量合同内比较 |

固定排除项：**不补 Crop seed43**。它不是本文结论的必要验证，也不用于构造三方法对称性。

## 3. 当前主结果：Original 必须在场

下表全部为当前选定 checkpoint 的 seed42 结果，单位为 `%`。Original 的精确七项均值为
`55.3556`，按既有测量合同报告为 `55.36`。

| Method | Macro* | Δ vs Original | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 55.3556 | — | 50.7853 | 59.0000 | 65.5556 | 48.1848 | 39.0300 | **74.3333** | 50.6000 |
| Crop S80 | 62.2288 | +6.8732 | **81.6754** | **74.5000** | 58.8889 | **55.3358** | 46.4684 | 67.3333 | 51.4000 |
| TGVF S64 | 59.8086 | +4.4531 | 74.3455 | 66.5000 | 65.5556 | 44.5446 | 44.9814 | 72.3333 | 50.4000 |
| Atomic S16 | **63.0827** | **+7.7271** | 71.7277 | 73.5000 | **66.1111** | 54.2720 | **51.3011** | 69.6667 | **55.0000** |

可直接写入正文的事实边界：

- 三个工具方法的 Macro* 都高于 Original；当前表中 Atomic S16 的 matched-prompt Macro* 最高，
  Crop S80 是按用户指定报告的 80-step 终点基线，不再使用 post-hoc 最优 S32。
- TGVF 不是全榜最优方法，因此文章不能写成“通用性能支配”。它相对 Original 的主要整体
  增益在 VStar、HRBench 和 MMMU，并在一组更细的关系/数学任务中形成集中优势。
- Atomic 与 Crop 的 Macro* 接近，但来源不同；这支持能力分解，不支持“Atomic 已严格优于
  Crop”或“Crop+TGVF 存在因果 synergy”。
- Original 在 MathVista 和部分视觉强度切片上仍优于工具方法，必须作为负面边界一起报告。

## 4. 用于彰显优势的 sub-benchmark 面板

### 4.1 预冻结选择规则

主图只使用官方 scorer 已提供、样本定义稳定且可对四种方法对齐的 sub-benchmark。候选项必须
满足以下至少一项：

1. 对应方法相对 Original 有正增益，且能映射到明确的视觉能力；
2. 对应方法在 Crop/TGVF/Atomic 中形成方法特异性领先；
3. 是会限制论文主张的重要反例。

不允许在 full-prompt 结果出来后重新挑选切片。主图展示精简面板，补充材料报告同一 family
的完整官方切片，避免只报有利项。

### 4.2 当前候选面板

| Sub-benchmark | n | Original | Crop S80 | TGVF S64 | Atomic S16 |
|---|---:|---:|---:|---:|---:|
| BLINK / Relative Depth | 30 | 83.33 | 83.33 | **86.67** | 80.00 |
| BLINK / Relative Reflectance | 30 | 50.00 | 30.00 | 46.67 | **70.00** |
| BLINK / Spatial Relation | 30 | **96.67** | 86.67 | 93.33 | 90.00 |
| HRBench / cross-image aggregate | 100 | 59.00 | 61.00 | 64.00 | **68.00** |
| HRBench / single-image aggregate | 100 | 59.00 | **88.00** | 69.00 | 79.00 |
| MathVista / arithmetic reasoning | 104 | 65.38 | 57.69 | **72.12** | 63.46 |
| MathVista / math word problem | 63 | 77.78 | 68.25 | **84.13** | 79.37 |
| MathVista / numeric commonsense | 36 | 47.22 | 44.44 | **58.33** | 50.00 |
| MathVista / visual question answering | 42 | 54.76 | 50.00 | **64.29** | 50.00 |
| OCR EN / text recognition | category | 60.49 | 70.23 | 55.05 | **73.38** |
| OCR CN / text recognition | category | 59.82 | **71.23** | 22.17 | 67.80 |
| MathVerse / Vision Only | 100 | 28.00 | 43.00 | 42.00 | **51.00** |
| MathVerse / Vision Intensive | 100 | **52.00** | 49.00 | 42.00 | 49.00 |

建议主图分成三块：

- **TGVF favorable regimes**：Relative Depth、MathVista arithmetic、math word、numeric
  commonsense、visual QA；
- **Atomic favorable regimes**：HR cross-image、Relative Reflectance、MathVerse Vision Only；
- **Boundary cases**：OCR CN text recognition、BLINK Spatial Relation、MathVerse Vision
  Intensive。

小样本切片（尤其 BLINK `n=30`）只作能力定位，必须同时给置信区间，不能单独承担文章核心结论。

## 5. 三项必须补齐的验证

### 5.1 RP67 image + D matched utility

目的：验证 RP67 提供的是 target-specific answer utility，而不只是额外 token 或任意 latent
扰动。

#### 既有 RP67 验证库存与口径纠正

必须先纠正一个容易混淆的历史口径：`POLICY_RL_SMALL_BATCH_PILOT_CLOSEOUT_20260814.md`
中 first-200 十臂表的 `image-only 72.5%`、`image + correct-D 83.5%`、zero/wrong
`76.0% / 77.0%` 来自 **RP66**，不能当作 RP67 结果。RP67 Step-2000 自身此前已经有以下
互补验证；它们不能简单相加，也不能把重复使用同一 867-row population 的结果当作独立复现。

| 层级 | RP67 既有验证 | 主要结果 | 能支持什么 | 关键边界 |
|---|---|---|---|---|
| 训练与表示健康 | 2,000-step 训练记录；main + 3 个 D-DeepStack branch | image-axis train top-1 `25% → 100%`，gap `−0.265 → 9.938`；内部评测全部 finite、collapse rate 0 | 优化闭合，D 没有数值崩溃或 token collapse | train 指标不是 held-out answer utility；S2000 validation 只有 8 rows |
| teacher-forced target retrieval | S2000 INT-DIAG，200 rows / 46 image groups | query top-1/top-2 `90% / 98%`，MRR `94.63%`；184 个完整控制样本中 correct-D 胜过全部 control `94.57%` | D 在固定 reader/readout 下具有很强的 image/target 可辨识性 | oracle target、teacher-forced NLL；不等价于自由生成或自主 target 选择 |
| image-conditioned answer utility | S2000 full-867 两臂 | image-only `644/867 = 74.28%`；image + correct-D `737/867 = 85.01%`，描述性差值 `+10.73 pp` | correct-D 在原图上下文上具有互补答题价值 | diagnostic semantic overlay；旧 artifact 未给 paired CI，且未含 zero/wrong-D |
| standalone / replacement 六臂 | S2000 first-200 | D-only correct/zero/wrong `36% / 34% / 37%`；direct replacement `35% / 21% / 32.5%` | 相对 zero 的 content signal 在 direct replacement 中存在 | D-only specificity 不成立；说明结论强烈依赖原图上下文，不能宣称 D 可替代图像 |
| earlier checkpoint wrong-image stress | S500 full-867 两臂 | image + correct-D `85.70%`；same-target wrong-image-D `2.19%` | D 明确绑定图像内容，错误图像 D 会强烈干扰 reader | 是 S500 而非所选 S2000；control 过强且主动破坏，只作补充诊断 |
| native causal/free-continuation | S2000 INT-DIAG | 9 对 continuation accuracy `50%`、expected-direction flip `55.56%`；36 对 target-presence actual-direction accuracy `0%`、continuation accuracy `2.78%` | 暴露 teacher-forced readout 到 native generation 的断层 | **负/未闭合证据**，不能作为 RP67 有效性的正面论据 |
| downstream external utility | Frozen-RP67 TGVF / Atomic 的 CoreDev-2511 | 已在主表及 sub-benchmark 表与 Original、Crop 同列 | 表示可嵌入端到端 policy 并在若干视觉推理 regime 形成优势 | 与 policy、prompt、训练 recipe 纠缠，不是 RP67 单变量消融 |

文章证据层级据此固定为：本次 full-867 三臂 paired utility 是 RP67 机制主证据；旧 full-867
correct-D vs image-only 与 INT-DIAG 是支持证据；first-200 六臂和 native continuation 作为
限制证据；CoreDev 结果负责外部效度。这样既能使用已有验证，也不会把 RP66 数值错归给 RP67，
或把 teacher-forced readout 误写成 autonomous tool-use 证明。

#### RP67 既有指标详细附录

训练与完成身份：RP67 Step-2000 run ID 为
`RP-67-QWEN3-INSTRUCT-REP-BALANCED-T1-IMAGE-AXIS-GROUNDED-2000-GPU01`，run identity SHA256
为 `0b53d04cf8e4c8b665e76279da1df8d1e6ebabee63318c644a3bff5bad099b44`，最终 Adapter
manifest SHA256 为 `2ea098967ba36671d6975a17e3830778d441149c27f5f80e43e78daf818933b1`。
训练日志共有 200 个每 10-step 记录，Step 10 与 Step 2000 的同批次训练指标如下；它们用于
确认优化方向与数值健康，不当作 held-out 泛化结果。

| train metric | Step 10 | Step 2000 |
|---|---:|---:|
| image-axis top-1 | 25.00% | 100.00% |
| image-axis score gap | −0.2649 | 9.9379 |
| image-axis loss | 0.8818 | 0.00029 |
| Matrix-CE | 1.3904 | 0.4155 |
| evidence `L_gen` | 5.0440 | 1.3605 |
| total loss | 7.3622 | 1.8532 |

Step-2000 周期性 validation snapshot 只有 8 rows / 2 image groups：Matrix-CE `0.2018`、
`L_gen 1.2983`、Norm `0.7245`、total `1.5726`。小规模 validation 只能作执行 gate；主要
held-out 表示证据来自下述固定 200-row INT-DIAG。

INT-DIAG 的 teacher-forced NLL 越低越好。correct-D mean NLL 为 `1.2694`；相对每个控制的
结果为：

| Control | available n | control mean NLL | correct-D win rate | correct-D mean advantage |
|---|---:|---:|---:|---:|
| target only | 200 | 4.1035 | 99.50% | 2.8341 |
| random D | 200 | 3.9661 | 100.00% | 2.6967 |
| wrong D, same image | 200 | 6.5265 | 95.00% | 5.2571 |
| wrong D, different image | 184 | 9.9689 | 100.00% | 8.6801 |

同一 INT-DIAG 的 46 个 image group / 200 rows query matrix 达到 top-1 `90.00%`、top-2
`98.00%`、MRR `94.625%`、mean diagonal gap `3.0327`。184 条具备全部控制的样本中，
correct-D 同时胜过所有控制为 `174/184 = 94.57%`。main D 与 layer 8/16/24 branches 的
joint finite rate 均为 100%，near-identical-token collapse rate 均为 0；D/source mean-token-
norm ratio 分别为 `1.327 / 2.872 / 2.734 / 2.543`。这些 norm 只用来排除崩溃，不把绝对
scale 大小解释成语义质量。

不同 checkpoint / injection context 的历史 semantic overlay 不能混成一张同质 ablation，
但并列表明了结论边界：

| checkpoint / context | correct | zero / image-only | matched wrong | 读法 |
|---|---:|---:|---:|---|
| S500 first-200, D-only | 48.50% | 35.00% | 47.00% | content `+13.5 pp`，specificity 仅 `+1.5 pp` |
| S500 first-200, direct replacement | 37.00% | 22.00% | 35.00% | content `+15 pp`，specificity 仅 `+2 pp` |
| S2000 first-200, D-only | 36.00% | 34.00% | 37.00% | content `+2 pp`，specificity `−1 pp` |
| S2000 first-200, direct replacement | 35.00% | 21.00% | 32.50% | content `+14 pp`，specificity `+2.5 pp` |
| S500 full-867, image + D | 85.70% | — | 2.19% wrong-image D | 强 image binding stress；不是 S2000 control |
| S2000 full-867, image + D / image-only | 85.01% | 74.28% | — | 描述性互补效应 `+10.73 pp` |
| S2000 full-867, 本次 paired 三臂 | 84.89% | 77.74% zero-D | 63.32% same-image wrong-target D | 当前机制主结果；paired CI 见下表 |

S500 → S2000 并没有在所有 free-generation context 上单调改善：尤其 D-only specificity 从
微弱正差变成 `−1 pp`。另一方面，S2000 的 teacher-forced retrieval/readout 极强，且原图 + D
三臂结果支持 content utility 与 specificity。这种不一致本身是重要发现：RP67 学到的 D 更适合
作为原图上的条件残差，不应描述成独立视觉替代物；teacher-forced 可读性也不能自动推出 native
free-generation 因果性。

native 诊断保留原始负结果：9 对 counterfactual 的 continuation accuracy `50.00%`、
expected-direction flip `55.56%`、healthy termination `77.78%`；36 对 target-presence 的
actual-direction accuracy `0%`、continuation accuracy `2.78%`、healthy termination `93.06%`。
这些小样本探针不推翻 image+D 主结果，但明确阻止“RP67 已在 native continuation 中稳定控制
答案方向”的表述。

权威 artifact 索引：

- 训练与 INT-DIAG：
  `artifacts/representation/RP-67-qwen3-instruct-balanced-t1-image-axis-grounded-2000-gpu01/`
  下的 `metrics.jsonl` 与 `int-diag-step2000.json`；
- S2000 全验证闭合 receipt：
  `artifacts/representation_experiments/image_axis_grounding/evaluation/rp67_step2000_all_validations_complete_v2.json`；
- S2000 first-200 六臂：
  `artifacts/representation_experiments/image_axis_grounding/evaluation/rp67_step2000_first200_6arm_semantic_20260801/summary.json`；
- S2000 full-867 image-only / correct-D：
  `artifacts/representation_experiments/image_axis_grounding/evaluation/rp67_step2000_full867_acc_main_semantic_v2_20260801/summary.json`；
- S500 first-200 六臂与 full-867 wrong-image stress：
  `artifacts/representation_experiments/image_axis_grounding/evaluation/rp67_step0500_first200_6arm_semantic_20260731/summary.json`
  和 `rp67_step0500_full867_2arm_semantic_20260731/summary.json`。

上述 free-generation accuracy artifact 的固定 `claim_scope` 均为
`diagnostic_semantic_overlay_not_formal_pilot`；正文或附录不得删去这一标签。

固定三臂：

| Arm | 定义 | 作用 |
|---|---|---|
| `image_correct_D` | 原图 + 正确 target 的 RP67 D | treatment |
| `image_target_zero_D` | 原图 + 同位置零 D | 测 D content utility 的 control |
| `image_matched_wrong_D` | 原图 + 同图、不同 target、答案安全 donor 的 D | 测 target specificity 的 control |

主要量：

- `Δcontent = Acc(image_correct_D) - Acc(image_target_zero_D)`；
- `Δspecificity = Acc(image_correct_D) - Acc(image_matched_wrong_D)`；
- 辅助报告 `Acc(wrong) - Acc(zero)`、paired wins/losses、95% CI。

当前状态：

- [x] 完整 867 样本预检；203 个 image group 均可构造 answer-safe wrong mapping；
- [x] 8-way 可恢复正式生成完成；867 × 3 = 2,601 条记录，无缺失或重复；
- [x] 对 deterministic unresolved 完成固定 Qwen2.5-72B blind semantic overlay；
- [x] 汇总三臂 accuracy、paired delta、bootstrap/Wilson CI。

结果：

| Arm | Correct / 867 | Accuracy | 95% Wilson CI |
|---|---:|---:|---:|
| `image_correct_D` | 736 | **84.89%** | [82.35, 87.12] |
| `image_target_zero_D` | 674 | 77.74% | [74.85, 80.38] |
| `image_matched_wrong_D` | 549 | 63.32% | [60.06, 66.46] |

| Paired contrast | Δ accuracy | 95% paired bootstrap CI | wins / losses | exact McNemar p |
|---|---:|---:|---:|---:|
| correct − zero (`Δcontent`) | **+7.15 pp** | [4.73, 9.57] | 92 / 30 | `1.65e-8` |
| correct − matched-wrong (`Δspecificity`) | **+21.57 pp** | [18.22, 24.91] | 222 / 35 | `1.95e-34` |
| matched-wrong − zero | −14.42 pp | [−17.99, −10.84] | 71 / 196 | `1.04e-14` |

解释：correct D 同时显著优于零 D 和同图错误 target 的 D，支持 RP67 observation 携带
**内容相关且 target-specific** 的 answer utility。wrong D 仍有 63.32% accuracy，说明原图和
语言先验仍能解出大量题目；但 wrong 显著低于 zero，也说明不匹配 D 会主动干扰 reader，不能
把 wrong arm 当作“无信息”基线。

评分边界：原始 deterministic scorer 对 verbose answer 保守地返回 unresolved，因此最终结果
使用固定哈希的 Qwen2.5-72B blind semantic overlay。共 2,055 个唯一 judge 请求，零重试、零
长度截断。项目 schema 将它标为 `diagnostic_semantic_overlay_not_formal_pilot`；正文必须称为
diagnostic semantic accuracy，并在附录同时给 deterministic lower bound 与完整 judge 合同。
三臂 deterministic strict lower bound 分别只有 `1.04% / 52.71% / 0.81%`；其巨大差异主要
反映 correct/wrong 输出远比 zero verbose、从而更常 unresolved，不能把这组三臂 lower bound
当成可比较的 accuracy 估计。

生成根目录：
`artifacts/representation_experiments/answer_utility/evaluation/rp67_step2000_full867_three_arm_20260826_v1/`

语义 overlay：
`artifacts/representation_experiments/answer_utility/evaluation/rp67_step2000_full867_three_arm_semantic_20260826_v1/`

### 5.2 冻结最佳 checkpoint 的 full-prompt 评测

目的：检查现有优势是否依赖过度简化的 matched prompt。

固定设计：

- Pure TGVF：S64；
- Atomic：S16；
- checkpoint 在看 full-prompt 结果前冻结，不允许重新选 step；
- 仍使用 CoreDev-2511、相同官方 scorer、temperature 1、seed42；
- prompt 详细规定 target 必须包含“看什么 + 提取什么证据/关系”，Atomic 还规定 bbox、关系实体
  覆盖、禁止猜测答案；
- 可见工具上限与运行时均为 6；成功 observation 回显 target；
- 这是同一 benchmark 上的 prompt-shift robustness，不是 held-out dataset confirmation。

当前状态：

- [x] 新增独立协议 `full_visual_tool_prompt_v5_instruct_cap6`；
- [x] TGVF/Atomic prompt bundle、system prompt、tool schema、target echo 和 RNG 协议均绑定 SHA256；
- [x] S64/S16 两份计划通过真实训练配置和冻结 RP67 绑定预检；
- [x] 相关协议与计划测试 18/18 通过；
- [x] 修复并回归测试已完成训练中“历史最优 checkpoint + 历史 RP67 manifest”的等待边界；
- [x] TGVF S64 与 Atomic S16 full-prompt 生成均完成并通过完整性核验：每臂 2,240 rows、
  2,240 个唯一 sample / trajectory / result identity、零重复；
- [x] TGVF S64 七套官方 scorer 全部完成：7/7 slices，summary contract 通过，judge parse
  failure 为 0；
- [x] Atomic S16 七套官方 scorer 全部完成：7/7 slices，summary contract 通过，judge parse
  failure 为 0；
- [x] 两臂完整结果与 matched prompt、Original、Crop 同表报告。

完成快照（2026-08-26 02:40 JST）：TGVF 与 Atomic 均为 `2,240 / 2,240`，两份
completion receipt 均已生成。TGVF stop 分布为 2,005 final-answer、156 direct-answer、78
max-tokens、1 call-cap，累计 41 次工具错误；Atomic 为 2,003 final-answer、225 direct-answer、
12 max-tokens，累计 10 次工具错误。错误计数保留为质量诊断，但不影响各自 2,240-row 唯一覆盖。

两条线路的 full-prompt 文章口径结果如下。Original 仍是 raw direct reference，不把它误写成
严格 paired prompt control；`Δ vs matched` 才是同 checkpoint、同任务和同 seed 下的 prompt-shift
稳健性量。

| Reference / run | Macro* | Δ vs matched | Δ vs Original | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original raw direct | 55.3556 | — | — | 50.7853 | 59.0000 | 65.5556 | 48.1848 | 39.0300 | **74.3333** | 50.6000 |
| Crop S80 / matched prompt | 62.2288 | reference | +6.8732 | **81.6754** | **74.5000** | 58.8889 | **55.3358** | 46.4684 | 67.3333 | 51.4000 |
| TGVF S64 / matched prompt | 59.8086 | reference | +4.4531 | 74.3455 | 66.5000 | 65.5556 | 44.5446 | 44.9814 | 72.3333 | 50.4000 |
| TGVF S64 / full prompt | 58.5138 | −1.2949 | +3.1582 | 71.7277 | 64.5000 | 66.1111 | 39.4659 | 45.7249 | 68.6667 | 53.4000 |
| Atomic S16 / matched prompt | **63.0827** | reference | **+7.7271** | 71.7277 | 73.5000 | **66.1111** | 54.2720 | **51.3011** | 69.6667 | 55.0000 |
| Atomic S16 / full prompt | 60.4684 | **−2.6142** | +5.1128 | 70.6806 | 67.0000 | 63.3333 | 50.4349 | 46.0967 | 70.3333 | **55.4000** |

full prompt 相对 matched prompt 的七项 delta 依次为 VStar `−2.6178`、HR `−2.0000`、
BLINK-180 `+0.5556`、OCR mean `−5.0787`、MMMU-269 `+0.7435`、MathVista
`−3.6667`、MathVerse `+3.0000 pp`。因此当前结论不是“prompt 完全无影响”，而是：TGVF
总体优势经较完整 prompt 后仍保留，Macro* 仍比 Original 高 `3.1582 pp`，七项中有五项高于
Original；但 prompt shift 造成 `1.2949 pp` 总体回落，损失主要集中在 OCR、MathVista 和
VStar。sub-benchmark 上，full prompt 仍保留 Relative Depth `83.33%`、HR cross aggregate
`62.00%`、MathVista numeric commonsense `55.56%` 和 visual QA `59.52%` 等定位，同时
Relative Reflectance 从 matched 的 `46.67%` 升至 `66.67%`；原先 arithmetic 和 math-word
优势分别回落至 `63.46% / 74.60%`。这些变化必须作为能力迁移而不是单一“更强/更弱”报告。

Atomic full prompt 的 Macro* 为 `60.4684`，比自身 matched prompt 低 `2.6142 pp`，但仍比
Original 高 `5.1128 pp`，并比 TGVF full prompt 高 `1.9546 pp`。七项相对自身 matched 的
delta 为 VStar `−1.0471`、HR `−6.5000`、BLINK-180 `−2.7778`、OCR mean
`−3.8370`、MMMU-269 `−5.2045`、MathVista `+0.6667`、MathVerse `+0.4000 pp`。
这说明组合工具在完整 prompt 下仍有总体价值，却不够稳定到成为文章核心：预冻结的 Atomic
favorable regimes 中，HR cross 从 `68.00%` 降到 `56.00%`，Relative Reflectance 从
`70.00%` 降到 `46.67%`；只有 MathVerse Vision Only 从 `51.00%` 保持到 `52.00%`，而原先的
负面边界 Vision Intensive 从 `49.00%` 回升到 `53.00%`。根据第 6 节事前门槛，Atomic 应作为
prompt-sensitive exploratory extension / appendix 结果，不作为主方法，也不声称稳定 synergy。

OCR 生成行为给出了一个额外的协议边界。600-row OCR 切片上，TGVF full prompt 的答案长度
p95 / p99 为 `24,324 / 89,469` 字符，Atomic 为 `3,849 / 19,641`；TGVF 的极长输出尾部在
完整 prompt 下没有被消除，并与其 OCR mean 下滑同时出现。该相关性只作 termination/verbosity
诊断，不能据此单独断言准确率下降的因果来源。

full-prompt 权威 summary：

- TGVF：`artifacts/policy/PRL-25-C-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-C-FROZEN-RP67-TFREE-TEACHER25-S64-FULL-PROMPT-V5-CAP6-COREDEV2511-SEED42-V1/step64/scoring/coredev-official-v1/coredev-2511-eval-summary.json`；
- Atomic：`artifacts/policy/PRL-25-D-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-teacher25-80step-ws8/evaluation/PRL25-D-ATOMIC-CROP-TGVF-RP67-TFREE-TEACHER25-S16-FULL-PROMPT-V5-CAP6-COREDEV2511-SEED42-V1/step16/scoring/coredev-official-v1/coredev-2511-eval-summary.json`。

计划文件：

- `configs/evaluation/prl25_c_frozen_rp67_tfree_teacher25_s64_full_prompt_v5_cap6_coredev2511_plan.json`
- `configs/evaluation/prl25_d_atomic_crop_tgvf_tfree_teacher25_s16_full_prompt_v5_cap6_coredev2511_plan.json`

### 5.3 Atomic 无偏 target 合格率审计

目的：把“协议可解析”与“语义 target 合格”分开，确定 Atomic 是否能在正文中被描述为稳定地产生
高质量 target。

固定抽样与盲审设计：

- matched prompt 与 full prompt 各抽取 200 条**实际使用工具**的 trajectory；
- 以 `dataset × arm` 分层，层内按 SHA256 排序抽样；每条 trajectory 只审第一次工具调用，防止
  多调用样本过度加权；
- 标注者不看到 arm 名、checkpoint、最终答案、正确性、reward 或 benchmark score；
- 两名独立标注者；分歧由第三人裁决；报告 agreement 和逐标准 Wilson CI。

逐条全部通过才计为合格：

1. referent 可识别；
2. 明确说明要取得的视觉证据/属性/关系，不只是物体名；
3. 不泄漏猜测答案或后续计算结果；
4. bbox 与 target 一致；
5. 关系/比较任务包含必要实体；
6. target 是视觉查询，而不是含糊的后续推理指令。

当前状态：

- [x] 审计 rubric 和正文纳入门槛冻结；
- [x] 在 matched S16 的 1,863 条工具 trajectory 上实测物化 200 条盲审样本；七个数据集按
  population-proportional quota 覆盖，review view 未泄漏 arm、dataset、sample ID、correct、
  reward、score 或 final answer；
- [x] 生成 matched/full 两个 blind audit pack：每臂 200，共 400 条；review view 仅含 bbox、
  call index、image path/hash、question、opaque review ID、schema 和 target；400 个 review ID
  唯一，未出现 arm、dataset、sample ID、correct、reward、score 或 final answer 字段；
- [ ] 双人盲标、裁决、Wilson CI 与 agreement；
- [ ] 根据审计结果界定 Atomic 探索性分析可使用的 target-quality 表述。

正式盲审根目录：
`artifacts/evaluation/neurips-workshop-atomic-target-audit-20260826-v1/`。manifest SHA256 为
`5af64da18e03d7455ab523e011887baf7d1361e50a59d19914f10380bbd165cc`；状态为
`ready_for_blind_annotation`。matched/full 可用工具 trajectory population 分别为
`1,863 / 2,009`，按各自七套数据 population-proportional 抽样 200 条。coordinator key 与
review view 分离；人工标注完成前不得报告 target 合格率或据此升级 Atomic 的正文地位。

## 6. Atomic 纳入正文的决策门槛

Atomic 进入正文核心方法必须同时满足：

1. full-prompt Macro* 和主要 Atomic favorable sub-benchmark 不发生足以推翻当前定位的崩塌；
2. target audit 的 all-pass rate 及逐标准结果可被透明报告；
3. 文章只声称观察到的任务条件优势，不声称未被单变量消融证明的 synergy；
4. Original、Crop、TGVF 和 Atomic 四列同时出现，且 MathVerse Vision Intensive 等负面切片保留。

如果任一条件不满足，Atomic 降级到 exploratory analysis / appendix。Pure TGVF + RP67 utility
仍作为主机制线，Crop 作为强基线。

**当前决策：第 1 项已不满足。** Atomic full-prompt Macro* 虽仍为 `60.4684`，但 HR cross 与
Relative Reflectance 两个预冻结 favorable regimes 分别下降 `12.00 / 23.33 pp`，足以推翻
“这些特异优势具有 prompt 稳健性”的定位。因此 Atomic 固定降级为 exploratory analysis /
appendix；后续 target audit 无论结果好坏，都只改变该探索性分析的解释边界，不再事后修改方法
层级。

## 7. Claim–evidence–boundary 台账

| Claim | 当前证据 | 必须保留的边界 | 状态 |
|---|---|---|---|
| TGVF 改善一组 target-conditioned reasoning regimes | Relative Depth；MathVista arithmetic、word、numeric、visual QA；逐题案例 | 不是总体最优；OCR 和精细像素读取弱；BLINK 切片小 | 可写，待 CI |
| RP67 D 具有内容 utility 和 target specificity | correct−zero `+7.15 pp`；correct−wrong `+21.57 pp`；两者 95% CI 不跨零 | diagnostic semantic overlay；oracle target 不测自主工具选择 | 已支持 |
| Atomic matched prompt 下在跨图、反射率和 Vision Only 上显示优势 | matched HR cross、BLINK reflectance、MathVerse Vision Only | full prompt 仅保留 Vision Only；target 合格率未闭合 | 探索性/附录，核心门槛未通过 |
| 详细 prompt 下工具总体增益仍存在 | TGVF / Atomic full-prompt Macro* 为 `58.5138 / 60.4684`，分别比 Original 高 `3.1582 / 5.1128 pp` | 相对各自 matched prompt 下降 `1.2949 / 2.6142 pp`；同数据 prompt shift，不是新 benchmark 泛化 | 已支持，带退化边界 |
| 工具方法整体优于 raw direct Original | 三个选定 checkpoint 的 Macro* 均高于 55.36 | Original 非 paired control；MathVista 等单项仍可能更强 | 可写 |

## 8. 论文实验部分建议结构

1. **Comprehensive comparison.** 四方法七套 benchmark 主表，Original 永不缺席。
2. **Where target-conditioned evidence helps.** 预冻结 sub-benchmark 图，突出 TGVF 的关系、
   深度和视觉数学优势，同时给负面切片。
3. **Does RP67 carry target-specific answer utility?** 867 样本 correct/zero/wrong 三臂与
   paired effect。
4. **Robustness to a complete tool-use prompt.** S64/S16 frozen-checkpoint prompt shift。
5. **Exploratory Atomic Crop+TGVF.** 固定为附录或正文探索性分析；target audit 决定可报告的
   target-quality 表述，不改变核心方法层级。
6. **Qualitative mechanisms and failures.** 复用已有真实 trajectory 与 bbox 案例，但把机制语言
   限定为 behavior-level inference。

## 9. 当前推进顺序

1. [已完成] RP67 semantic overlay 和 CI，机制主张已锁定；
2. [已完成] 用 8 GPU 依次运行 TGVF S64、Atomic S16 full prompt，并闭合七项官方评分；
3. [已完成] full inference 闭合后，从 matched/full inference JSONL 物化正式 Atomic
   blind audit pack；
4. [full prompt 已回填；audit 待人工标注] 将 full-prompt 和 target audit 写回本文件；
5. [待写作] 形成英文 Experiments/Discussion 初稿；
6. [明确不做] Crop seed43。

## 10. 证据来源

- Original 定义和 Macro* 合同：
  `docs/POLICY_RL_COREDEV2511_MEASUREMENT_CONTRACT_AND_BASELINES_20260812.md`
- 80-step 三线路数值与 checkpoint 选择：
  `docs/PRL25_BS16_TEACHER25_80STEP_PHASE3_PLAN_20260820.md`
- 三方法逐题轨迹与失败案例：
  `docs/PRL25_CROP_TGVF_ATOMIC_QUALITATIVE_CASE_ANALYSIS_20260825.md`
- Crop S80 官方 summary：
  `artifacts/policy/PRL-25-B-qwen3-instruct-full-crop-exact-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-B-CROP-EXACT-COREDEV2511-STEP80-TEMP1-SEED42-UNIFIED-V1/step80/scoring/coredev-official-v1/coredev-2511-eval-summary.json`
- TGVF S64 / Atomic S16：对应 six-point evaluation 的 `step64` / `step16` 官方 summary 与
  `paired-summary.json`。

注意：主仓当前 qualitative 文档可能尚未进入本 worktree 的提交历史；本文只把它作为只读证据
来源，不覆盖主仓未提交内容。
