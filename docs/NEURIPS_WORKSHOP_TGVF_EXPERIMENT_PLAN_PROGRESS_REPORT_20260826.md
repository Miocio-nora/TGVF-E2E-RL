# NeurIPS Workshop：TGVF 文章实验计划、推进台账与阶段报告

更新时间：2026-08-26（Asia/Tokyo）

状态：**实验进行中；RP67 三臂验证已闭合；广义 full-prompt stress test 与严格 target-only
matched prompt 两臂均已完成并评分；只增加 target 定义与案例后，TGVF S64 / Atomic S16
Macro* 仍高于 Original，但相对各自 matched prompt 有温和回退。No-Tool RL 因果对照已冻结
为 32-step 正式主终点，保留 S0/S8/S16/S32；独立实现、严格配置与 CPU 回归已完成，真实
1-step canary 已通过，正式 8 卡训练已闭合 S32，S8/S16/S32 三个永久 checkpoint 均已验收。
下一步完成 matched no-tool / raw-direct 双协议评测、
正式 Atomic matched/target-only 盲审与调用行为对照。**

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
raw direct 端到端参考。**Atomic Crop+TGVF** 目前仍列探索性扩展；其正文层级等待真正的
target-only 稳健性与无偏 target 合格率审计。已完成的广义 full-prompt stress test 同时改变了
多项 prompt/observation 合同，不能单独用于决定 Atomic 的正文层级。

## 2. 固定术语和比较口径

| 简称 | 本文固定含义 | checkpoint / run | 解释边界 |
|---|---|---|---|
| **Original** | 原始 Qwen3-VL-8B-Instruct；无视觉工具、无自定义 system prompt | `PRL-04-R2-raw-instruct-coredev2511-gpu4567-r4` | 必须进入所有主表和 sub-benchmark 表；因 prompt/agent protocol 不同，只是端到端 direct reference，不是严格 paired control |
| **No-Tool RL** | 同一 Qwen3-VL-8B-Instruct 做 full-model RL，但没有 Crop、TGVF、RP67、工具 schema 或工具调用 | `PRL-25-F-...-NO-TOOL-RL-...-32STEP-WS8`；S32 为事前冻结主终点 | 用于回答增益有多少来自 RL 本身；matched no-tool 为主要因果对照，raw-direct transfer 为诊断，不得改称 Original |
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

当前面板是根据 matched-prompt 结果形成的**探索性解释面板**，不是事前注册的 confirmatory
endpoint。本文在 target-only 结果揭晓前冻结其 v2 版本；此后不因 target-only 结果重选切片。
正文展示精简优势面板，补充材料同时报告同一 family 的完整切片，避免只报有利项。

### 4.2 正文候选：扩展优势面板 v2

表中粗体表示四种方法的行最优；最后一列是 `max(TGVF, Atomic) − Original`，因此即使 Crop
是行最优，也能看清两条 target-conditioned 方法相对 Original 的增益。

| Sub-benchmark | n | Original | Crop S80 | TGVF S64 | Atomic S16 | Best TGVF/Atomic Δ |
|---|---:|---:|---:|---:|---:|---:|
| VStar / direct attributes | 115 | 48.70 | **83.48** | 70.43 | 69.57 | **+21.74** |
| VStar / relative position | 76 | 53.95 | 78.95 | **80.26** | 75.00 | **+26.32** |
| HRBench / cross-image aggregate | 100 | 59.00 | 61.00 | 64.00 | **68.00** | **+9.00** |
| HRBench / single-image aggregate | 100 | 59.00 | **88.00** | 69.00 | 79.00 | **+20.00** |
| BLINK / Counting | 30 | 66.67 | 70.00 | **73.33** | **73.33** | **+6.67** |
| BLINK / Object Localization | 30 | 56.67 | 56.67 | 70.00 | **73.33** | **+16.67** |
| BLINK / Relative Depth | 30 | 83.33 | 83.33 | **86.67** | 80.00 | **+3.33** |
| BLINK / Relative Reflectance | 30 | 50.00 | 30.00 | 46.67 | **70.00** | **+20.00** |
| MathVista / numeric commonsense | 36 | 47.22 | 44.44 | **58.33** | 50.00 | **+11.11** |
| MathVista / arithmetic reasoning | 104 | 65.38 | 57.69 | **72.12** | 63.46 | **+6.73** |
| MathVista / visual question answering | 42 | 54.76 | 50.00 | **64.29** | 50.00 | **+9.52** |
| MathVista / math word problem | 63 | 77.78 | 68.25 | **84.13** | 79.37 | **+6.35** |
| OCR EN / text recognition | category | 60.49 | 70.23 | 55.05 | **73.38** | **+12.89** |
| OCR EN / visual text understanding | category | 75.00 | 81.61 | 80.00 | **83.21** | **+8.21** |
| OCR CN / visual text understanding | category | 35.00 | 60.00 | **65.00** | 55.00 | **+30.00** |
| MathVerse / Text Lite | 100 | 53.00 | 56.00 | 58.00 | **59.00** | **+6.00** |
| MathVerse / Vision Only | 100 | 28.00 | 43.00 | 42.00 | **51.00** | **+23.00** |

建议主图分成三块：

- **target-conditioned shared gains**：VStar、HR、Counting、Object Localization；
- **TGVF-concentrated gains**：relative position/depth、MathVista 四项、OCR CN visual text
  understanding；
- **Atomic-concentrated gains**：HR cross、Relative Reflectance、OCR EN text recognition、
  MathVerse Vision Only。

正文同时保留一个小型 boundary companion panel：BLINK IQ Test、Spatial Relation、MathVista
geometry reasoning、MathVerse Vision Intensive。这样主图可以彰显优势，但不会暗示全面支配。

### 4.3 MathVista MINI 低于 Original 的归因

这不是测试 subset 不同造成的。四种方法都在同一份完整 `MathVista_MINI` 上评分，逐题核对后
均为相同的 300 个 `index`；Original 答对 223 题，而 Crop、TGVF、Atomic 分别答对
202、217、209 题。按官方判分函数逐题配对得到：

| Method | Correct / 300 | Δ correct vs Original | Gained: method correct, Original wrong | Lost: Original correct, method wrong |
|---|---:|---:|---:|---:|
| Original | 223 | — | — | — |
| Crop S80 | 202 | -21 | 15 | 36 |
| TGVF S64 | 217 | -6 | 26 | 32 |
| Atomic S16 | 209 | -14 | 19 | 33 |

对 TGVF，5 个互斥官方 task 的净变化为：textbook question answering `-7` 题、figure
question answering `-4`、geometry problem solving `-3`，被 math word problem `+4` 和
visual question answering `+4` 部分抵消，合计 `-6`。按互斥 question type 看，TGVF 在
free-form 上 `+1` 题，在 multi-choice 上 `-7` 题。因此当前证据更准确的表述是：TGVF 在
视觉数学中的 word problem / visual QA 子域有集中收益，但没有保住 Original 在 textbook、
figure QA、geometry solving 和多选题上的优势；不能归结为一个更难的工具方法专属 subset。

Appendix A 完整列出各 benchmark 的 aligned sub-benchmark。MathVista 的 12 行是官方 scorer
输出的 5 个互斥 task 加 7 个可重叠 skill；重叠 skill 的增减不能相加来解释总体 6 题差值。
当前逐题数据只能定位回退发生在哪里，尚不能在没有单变量消融时断言原因是工具干扰、prompt
shift 或答案格式。

### 4.4 三种工具方法的调用覆盖率与调用强度

#### 4.4.1 统计口径与整体结果

调用统计来自三个 matched-prompt 最佳 checkpoint 的 `rank-0..3.jsonl` trajectory。三者在完全
相同的 `2,240` 个受支持单图 ID 上各有一条记录，其中 BLINK 使用共同单图 `n=180`，
MMMU-Pro 使用共同单图 `n=269`。本文区分以下口径：

- **attempted question**：`tool_calls + tool_errors > 0`，即模型至少生成过一次工具尝试；
- **successful-use question**：`tool_calls > 0`，即至少一次调用被执行并返回 observation；
- **executed calls**：trajectory 中 `tool_calls` 的总数；
- **invalid attempts**：trajectory 中 `tool_errors` 的总数，不计入有效工具调用；
- **repeat-use question**：有效调用次数至少为 2。

Original 没有工具接口；下表的 Original 行是在同一 `2,240` 个 ID 上按协议定义的零调用参照，
不是从 Original trajectory 反推的统计。Atomic 的 `tgvf_crop_tool` 是一次调用内联合产生
Crop 与 TGVF observation 的原子工具，不能拆成两次工具调用。

| Method | Tool function | Attempted questions | Successful-use questions | Executed calls | Invalid attempts | Execution yield | Calls/question | Calls/used question | Repeat-use questions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | none | 0 (0.00%) | 0 (0.00%) | 0 | 0 | — | 0.000 | — | 0 (0.00%) |
| Crop S80 | `image_zoom_in_tool` | 1,576 (70.36%) | 1,502 (67.05%) | 1,508 | 100 | 93.78% | 0.673 | 1.004 | 3 (0.13%) |
| TGVF S64 | `tgvf_focus_tool` | **2,012 (89.82%)** | **2,010 (89.73%)** | 2,011 | 3 | **99.85%** | 0.898 | 1.000 | 1 (0.04%) |
| Atomic S16 | `tgvf_crop_tool` | 1,866 (83.30%) | 1,863 (83.17%) | **2,300** | 36 | 98.46% | **1.027** | **1.235** | **221 (9.87%)** |

`Execution yield = executed calls / (executed calls + invalid attempts)`。三个方法的 `1,508`、
`2,011`、`2,300` 次 executed calls 均产生了 observation；小于 100% 的 execution yield 来自
另外记录的无效尝试，而不是已执行工具返回失败。

#### 4.4.2 不同 benchmark set 的调用覆盖率

每格为 `successful-use questions / n (rate)`。该表回答“模型在多少题上实际使用了工具”，不把
无效尝试算作成功使用。

| Set | n | Crop S80 | TGVF S64 | Atomic S16 |
|---|---:|---:|---:|---:|
| VStarBench | 191 | 74/191 (38.74%) | 186/191 (97.38%) | **188/191 (98.43%)** |
| HRBench4K | 200 | 104/200 (52.00%) | **199/200 (99.50%)** | 198/200 (99.00%) |
| BLINK single-image | 180 | 144/180 (80.00%) | **178/180 (98.89%)** | **178/180 (98.89%)** |
| OCRBench v2 | 600 | 444/600 (74.00%) | 556/600 (92.67%) | **560/600 (93.33%)** |
| MMMU-Pro single-image | 269 | 177/269 (65.80%) | 232/269 (86.25%) | **243/269 (90.33%)** |
| MathVista MINI | 300 | 233/300 (77.67%) | **268/300 (89.33%)** | 222/300 (74.00%) |
| MathVerse MINI | 500 | 326/500 (65.20%) | **391/500 (78.20%)** | 274/500 (54.80%) |

#### 4.4.3 不同 benchmark set 的调用次数与频率

每格为 `executed calls / calls per eligible question`。分母始终是该 set 的全部受支持题目，而
不是只包含工具调用题，因此不同方法可直接比较调用强度。

| Set | Crop S80 | TGVF S64 | Atomic S16 |
|---|---:|---:|---:|
| VStarBench | 74 / 0.387 | 186 / 0.974 | **191 / 1.000** |
| HRBench4K | 104 / 0.520 | 199 / 0.995 | **218 / 1.090** |
| BLINK single-image | 144 / 0.800 | 178 / 0.989 | **307 / 1.706** |
| OCRBench v2 | 450 / 0.750 | 556 / 0.927 | **728 / 1.213** |
| MMMU-Pro single-image | 177 / 0.658 | 232 / 0.862 | **287 / 1.067** |
| MathVista MINI | 233 / 0.777 | 268 / 0.893 | **281 / 0.937** |
| MathVerse MINI | 326 / 0.652 | **392 / 0.784** | 288 / 0.576 |

#### 4.4.4 每题有效调用次数分布与无效尝试

| Method | 0 calls | 1 call | 2 calls | 3 calls | 4 calls | 5+ calls |
|---|---:|---:|---:|---:|---:|---:|
| Original | 2,240 (100.00%) | 0 | 0 | 0 | 0 | 0 |
| Crop S80 | 738 (32.95%) | 1,499 (66.92%) | 2 (0.09%) | 0 | 0 | 1 (0.04%) |
| TGVF S64 | 230 (10.27%) | 2,009 (89.69%) | 1 (0.04%) | 0 | 0 | 0 |
| Atomic S16 | 377 (16.83%) | 1,642 (73.30%) | 120 (5.36%) | 44 (1.96%) | 23 (1.03%) | 34 (1.52%) |

无效尝试的错误构成也不同：Crop S80 的 100 次包括 `invalid_crop=84`、`context_limit=16`；
TGVF S64 只有 `tool_parse.invalid_tool_name=3`；Atomic S16 的 36 次包括
`tool_call_limit_exceeded=21`、`tool_parse.invalid_bbox=8`、
`tool_parse.incomplete_tool_call=5`、`tool_parse.invalid_tool_name=2`。它们必须与 executed
calls 分开报告，因为错误尝试没有产生工具 observation。

#### 4.4.5 可写结论与边界

- Crop S80 表现为**选择性、近乎单次**调用：整体成功使用率 `67.05%`，`99.80%` 的工具使用题
  只有一次有效调用；其覆盖率在 BLINK 和 MathVista 较高，在 VStar 最低。
- TGVF S64 表现为**高覆盖、近乎固定一次**调用：整体成功使用率 `89.73%`，只有 1 题发生
  两次有效调用；VStar、HRBench 和 BLINK 的覆盖率均超过 `97%`。
- Atomic S16 表现为**按 set 改变调用强度**：成功使用率 `83.17%`，但 `9.87%` 的全部题目
  出现重复调用，使整体达到 `1.027 calls/question`；重复检索主要集中在 BLINK 和 OCR。
- 调用率和调用次数是行为描述，不是因果 utility。不能由“调用更多”直接推出“工具更有效”；
  后续应按相同 ID 报告 `0/1/2+` 调用组的正确率与置信区间，并明确其受 policy 自选择混杂。

target-only 版本完成后，必须用同一定义追加一张对照表；在此之前不得把 matched-prompt 与
target-only 的调用统计合并。

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

### 5.2 冻结最佳 checkpoint 的 prompt 补充评测

目的：检查现有优势是否依赖过度简化的 matched prompt。

#### 5.2.1 已完成的广义 full-prompt stress test

2026-08-26 的逐字节协议审计发现，matched prompt 本身已经要求 `<think>...</think>`、按需
tool call 和 plain-text final；已完成的 `full_visual_tool_prompt_v5_instruct_cap6` 不只细化
target，还改变了逐轮 reasoning/final-only 指令、native schema 注入方式，并在成功 observation
中回显 target。因此下述结果保留为有效的**广义 prompt bundle stress test**，但不得写成“只增加
target 定义”的单变量结果，也不得把下降归因于 target 细化。

该 stress test 的固定设计：

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
这说明组合工具在广义完整 prompt 下仍有总体价值，但不能由此单独决定其正文层级：预冻结的 Atomic
favorable regimes 中，HR cross 从 `68.00%` 降到 `56.00%`，Relative Reflectance 从
`70.00%` 降到 `46.67%`；只有 MathVerse Vision Only 从 `51.00%` 保持到 `52.00%`，而原先的
负面边界 Vision Intensive 从 `49.00%` 回升到 `53.00%`。这些数值支持 prompt sensitivity，
但不能替代下一小节已经闭合的 target-only 单变量评测。

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

#### 5.2.2 主要验证：target-only matched prompt

用户原定改动只有一项：补充 target 的详细定义和使用案例。新协议
`target_detailed_matched_prompt_v1` 因此冻结以下最小差异合同：

- 逐字保留 matched 的 `USER_PROMPT_V2`，包括已有的 `<think>...</think>`、按需 tool call 和
  plain-text final；不增加逐轮 reasoning 或分类别 final-only 规则；
- 保留同一 tool name、可执行 schema、Hermes call frame、`tools=[]` chat-template 路径、6 次
  运行上限和 matched success renderer；observation **不**回显 target；
- 仅在 matched system prompt 的 `<tools>` 与调用格式说明之间插入一个 target guide：定义
  target 必须同时包含 referent/entity 与所需视觉证据，并给出 3 个 valid 和 3 个 invalid 示例；
- checkpoint、CoreDev-2511、official scorer、temperature 1 和 seed42 不变；seed namespace
  逐字复用原 matched evaluation，使每题每轮随机流与原 matched run 对齐；
- S64/S16 在查看结果前冻结，不重新选 step。

实现与状态：

- [x] target-only prompt identity、system/user SHA256、protocol SHA256 与差异白名单冻结；
- [x] 81 项相关测试通过；白名单证明 matched system prompt 只有一个连续 target-guide 插入块，
  user suffix 完全相同，且不存在 full prompt 新增的逐轮/final-only 文本；
- [x] 两份真实 snapshot 预检通过：TGVF/Atomic 分别使用 matched 的
  `render_qwen_native_matched_tgvf_success_environment_text` /
  `render_qwen_native_matched_crop_tgvf_success_environment_text`；
- [x] 两臂均完成 2,240 条共同支持单图推理；七套官方 scorer 均通过，共覆盖各自完整的
  2,511-row CoreDev 输入合同；TGVF / Atomic judge parse failure 为 `4 / 2`，均按固定
  deterministic-incorrect 策略处理且低于阈值；
- [x] 与 matched、广义 stress、Crop S80、Original 同表回填。

| Arm | Macro* | Δ vs own matched | Δ vs Original | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 55.3556 | — | — | 50.7853 | 59.0000 | 65.5556 | 48.1848 | 39.0300 | **74.3333** | 50.6000 |
| Crop S80 / matched | 62.2288 | reference | +6.8732 | **81.6754** | **74.5000** | 58.8889 | **55.3358** | 46.4684 | 67.3333 | 51.4000 |
| TGVF S64 / matched | 59.8086 | reference | +4.4531 | 74.3455 | 66.5000 | **65.5556** | 44.5446 | 44.9814 | 72.3333 | 50.4000 |
| TGVF S64 / target-only | 58.1788 | -1.6298 | +2.8233 | 75.3927 | 61.5000 | 63.8889 | 40.1453 | 45.7249 | 68.0000 | 52.6000 |
| Atomic S16 / matched | 63.0827 | reference | +7.7271 | 71.7277 | 73.5000 | 66.1111 | 54.2720 | 51.3011 | 69.6667 | 55.0000 |
| Atomic S16 / target-only | 60.8253 | -2.2574 | +5.4697 | 71.7277 | 70.5000 | 61.6667 | 49.3089 | 46.8401 | 69.3333 | **56.4000** |

target-only 改动没有带来总体提升，但也没有抹去相对 Original 的整体优势。TGVF 的主要回退在
HR `-5.00 pp`、OCR mean `-4.40 pp` 和 MathVista `-4.33 pp`，同时 VStar、MMMU-269、
MathVerse 分别提高 `+1.05 / +0.74 / +2.20 pp`。Atomic 的主要回退在 OCR mean `-4.96 pp`、
MMMU-269 `-4.46 pp`、BLINK-180 `-4.44 pp` 和 HR `-3.00 pp`，MathVerse 提高 `+1.40 pp`。
因此“更详细 target 定义本身普遍提高准确率”不被支持；可以支持的是两种方法在该单变量 prompt
干预下仍保持高于 Original 的 Macro*，同时存在明确的 benchmark-specific sensitivity。

计划文件：

- `configs/evaluation/prl25_c_frozen_rp67_tfree_teacher25_s64_target_detailed_matched_v1_coredev2511_plan.json`
- `configs/evaluation/prl25_d_atomic_crop_tgvf_tfree_teacher25_s16_target_detailed_matched_v1_coredev2511_plan.json`

权威 summary：

- TGVF：`artifacts/policy/PRL-25-C-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-C-FROZEN-RP67-TFREE-TEACHER25-S64-TARGET-DETAILED-MATCHED-V1-COREDEV2511-SEED42-V1/step64/scoring/coredev-official-v1/coredev-2511-eval-summary.json`；
- Atomic：`artifacts/policy/PRL-25-D-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-teacher25-80step-ws8/evaluation/PRL25-D-ATOMIC-CROP-TGVF-RP67-TFREE-TEACHER25-S16-TARGET-DETAILED-MATCHED-V1-COREDEV2511-SEED42-V1/step16/scoring/coredev-official-v1/coredev-2511-eval-summary.json`。

### 5.3 Atomic 无偏 target 合格率审计

目的：把“协议可解析”与“语义 target 合格”分开，确定 Atomic 是否能在正文中被描述为稳定地产生
高质量 target。

固定抽样与盲审设计：

- matched prompt 与 target-only prompt 各抽取 200 条**实际使用工具**的 trajectory；广义
  full-prompt stress arm 只作附加诊断；
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
- [x] 既有 matched/广义-full blind audit pack 已生成：每臂 200，共 400 条；review view 仅含 bbox、
  call index、image path/hash、question、opaque review ID、schema 和 target；400 个 review ID
  唯一，未出现 arm、dataset、sample ID、correct、reward、score 或 final answer 字段；
- [ ] 基于已闭合的 target-only inference 物化新的 matched/target-only 正式盲审包；既有
  广义-full pack 不替代该主要审计；
- [ ] 双人盲标、裁决、Wilson CI 与 agreement；
- [ ] 根据审计结果界定 Atomic 探索性分析可使用的 target-quality 表述。

正式盲审根目录：
`artifacts/evaluation/neurips-workshop-atomic-target-audit-20260826-v1/`。manifest SHA256 为
`5af64da18e03d7455ab523e011887baf7d1361e50a59d19914f10380bbd165cc`；状态为
`ready_for_blind_annotation`。matched/full 可用工具 trajectory population 分别为
`1,863 / 2,009`，按各自七套数据 population-proportional 抽样 200 条。coordinator key 与
review view 分离；人工标注完成前不得报告 target 合格率或据此升级 Atomic 的正文地位。

### 5.4 No-Tool RL：RL 本身的正式对照

这条新线路回答一个独立于工具机制的问题：当前 Crop、TGVF 和 Atomic 相对 raw Original 的
增益中，有多少可以由**同等训练预算下的 full-model RL 本身**解释。为避免把 raw base 与训练后
模型混为一谈，固定名称为 **No-Tool RL**；`Original` 始终只表示未经本文 RL 的原始
Qwen3-VL-8B-Instruct。

#### 5.4.1 事前冻结的训练合同

| 项目 | 冻结设置 |
|---|---|
| 正式主终点 | **S32**；不根据中途结果延长或改选更优 step |
| 保存与评测点 | S0、S8、S16、S32 |
| base / update | 与 PRL25 三条工具线相同的 Qwen3-VL-8B-Instruct；full-model Qwen update |
| 数据顺序 | 同一不可变 Teacher25 schedule `PRL22-TEACHER25-MIXED-SCHEDULE-v1` 的前 `32 × 16 = 512` 个 prompt |
| rollout 预算 | global batch 16、每 prompt 16 rollouts，共 `8,192` 个 rollout |
| 优化设置 | seed42、temperature 1、learning rate `1e-6`，其余 optimizer / GRPO 设置与 PRL25 对齐 |
| 奖励 | 保留相同 answer judge / correctness reward；无 TGVF utility reward；工具与重复调用惩罚因无工具而恒为零 |
| 明确移除 | Crop、TGVF、RP67 representation、tool schema、tool observation、tool agent loop |

No-Tool 的 visual-row 训练 prompt 只保留图像、canonical question、`<think>...</think>` 推理要求
和 plain-text final 要求；不加入工具定义、target 定义、bbox 案例或工具使用指令。ThinkLite 行保持
Teacher25 各条线路已有的原始 no-tool 合同不变。任何结构化工具调用文本都按 protocol error
处理，不能被当作有效动作。

#### 5.4.2 固定评测矩阵

每个 S0/S8/S16/S32 checkpoint 都执行两套评测，且在 S32 结果揭晓后不得重选协议：

1. **matched no-tool（主要）**：使用与 No-Tool RL visual-row 训练一致的 no-tool prompt，作为
   `No-Tool RL ↔ Crop/TGVF/Atomic RL` 的主要训练匹配对照；
2. **raw direct（诊断）**：使用 Original 的 raw direct prompt，衡量 RL 后能力能否迁移回原始
   Qwen 使用方式，并直接连接 `Original ↔ No-Tool RL`。

两套协议均使用同一 CoreDev2511 七项、Macro*、完整 aligned sub-benchmark 和逐 set 输出；
No-Tool RL 的工具调用率定义上应为 0，另做结构化工具文本泄漏审计。S32 是唯一正式 headline，
S8/S16 只呈现学习动态，不能用于 post-hoc checkpoint 选择。

#### 5.4.3 可支持的结论与边界

- 若工具方法超过 matched No-Tool RL，可把差额解释为与工具化训练合同相关的增益候选；只有在
  其他合同严格一致且置信区间支持时，才进一步归因到视觉工具。
- 若 No-Tool RL 已解释大部分相对 Original 的增益，正文必须把“RL 本身”列为主要替代解释，
  不能把全部增益归给 TGVF/RP67。
- matched no-tool 去除了工具 schema，因此它是当前最强的 RL-only control，但仍不可消除
  “存在工具 schema / agent loop”这一协议差异；raw-direct arm 只诊断迁移，也不替代 matched
  因果对照。

当前状态：

- [x] 名称、S32 主终点、训练预算、双评测协议和解释边界已事前冻结；
- [x] 独立干净分支上的 no-tool prompt/data route、空工具 schema、run config、S32 守护脚本与
  CPU compose/回归测试；
- [x] 真实 1-step canary：首轮在 Step 0 前发现共享 termination builder 仍把
  `</tool_call>` stop 当作全路径硬条件；修复后 canary 已闭合 Step 1 且零工具调用；
- [x] 正式 32-step 训练：S8/S16/S32 永久 checkpoint 与最终 supervisor acceptance 均已闭合；
- [ ] S0/S8/S16/S32 matched no-tool / raw-direct 双协议评测；
- [ ] 结果回填主表、sub-benchmark、调用行为表和 claim ledger。

执行与审计快照（2026-08-26 07:18 JST）：

- 正式 run ID：
  `PRL-25-F-QWEN3-INSTRUCT-FULL-NO-TOOL-RL-BS16-N16-TFREE-TEACHER25-32STEP-WS8`；
  canary 为独立 `BS4 × n2 / world4 / 1-step` 功能门，不进入正式 lineage。
- 执行分支 `neurips-notool-rl-s32` 已 push 到远端；当前授权恢复 commit 为
  `7645fe4a5bb095cb9e0c9cbb9f428abd98ef9aae`，核心 termination 修复 commit 为
  `8140ce05b2fbb6d5c9c9fd0594eb9f5b5984f6aa`。
- 回归证据：no-tool protocol/launcher/runtime/vLLM 相关测试 `129 passed`，native-agent-loop
  补充测试 `14 passed`；formal 与 canary CPU compose 均通过。
- 首轮失败发生在 rollout 前，因此没有污染恢复 canary。修复把 tool-boundary interpretation
  显式关闭于 No-Tool schema：工具样式文本仍完整保留并由 direct-only loop 判为 protocol error，
  既不解析/执行，也不让单条坏输出击穿整次运行。
- 恢复 canary 在 `137.21 s` 内完成 Step 1：4 prompts、8 trajectories、658 generated policy
  tokens；`successful_tgvf_observations=0`、`tool_call_attempt_rate=0.0`、
  `mean_tool_call_attempts=0.0`。world4 rolling checkpoint 的 model/optimizer/extra-state 四组 shards、
  paired project state、tracker 与 metrics 均已验收。canary 模式按框架合同不生成 permanent 副本。
- formal S32 supervisor 已通过 canary gate 与最终 CPU preflight，并在 tmux
  `prl25_f_no_tool_s32` 启动 fresh-S0 训练；正式运行使用 8 卡并保存、逐一验收 S8/S16/S32 的
  world8 模型、优化器和 extra-state shards。W&B 凭证缺失时只切换为本地 offline telemetry，
  不改变训练合同。
- 2026-08-26 07:23 JST 实时节点：world8 FSDP 与 8 个 vLLM 实例已完成初始化，trainer 显示
  `Training Progress 0/32`，首个 BS16 × n16 rollout batch 正在执行；tmux 存活，8 卡均已被该
  正式任务占用，日志中没有 traceback。此时 tracker/metrics 尚未出现是因为 Step 1 仍未提交，
  不能把 `0/32` 误写成已完成一个 optimizer step。
- 2026-08-26 07:44 JST 实时节点：正式训练已提交 **Step 2/32**，累计 32 prompts、512
  trajectories、217,722 generated policy tokens；累计 mean answer reward 为 `0.6855`，format
  error rate 为 `0.2266`。按 no-tool 合同，`successful_tgvf_observations=0`、
  `tool_call_attempt_rate=0.0`、`mean_tool_call_attempts=0.0`。Step 1/2 分别耗时约
  `626.8 / 592.2 s`；Step 2 rolling checkpoint 已验收 8 份 model、8 份 optimizer 与 8 份
  extra-state shard。tmux/supervisor 存活且无 traceback；采样时 GPU 正处于 checkpoint 写盘后的
  短暂空闲窗口。按当前均速粗估，S8 约在 1 小时后、S32 约在 5 小时后到达；该估时不作为实验结果。
- 2026-08-26 08:52 JST 关键节点：正式训练已提交并验收首个保留点 **S8/32**。累计
  128 prompts、2,048 trajectories、928,070 generated policy tokens；累计 mean answer reward
  为 `0.5962`，format error rate 为 `0.2378`，这些是在线训练遥测而非 benchmark 结果。
  No-Tool 合同持续满足：`successful_tgvf_observations=0`、`tool_call_attempt_rate=0.0`、
  `mean_tool_call_attempts=0.0`。S8 rolling checkpoint 与 permanent checkpoint 均存在；后者含
  permanent receipt、8 份 model、8 份 optimizer、8 份 extra-state shard 及配对 project state，
  共 40 个文件。tmux/supervisor 存活且无 traceback，训练已继续向 S16/S32 推进；按当前约
  `11.1 min/step` 的运行均速粗估，S16 约还需 1.5 小时、S32 约还需 4.5 小时。
- 2026-08-26 13:19 JST 完成节点：正式训练已闭合 **S32/32**，supervisor 写入
  `step32-accepted` 后正常退出。最终累计 512 prompts、8,192 trajectories、2,826,178
  generated policy tokens；累计 mean answer reward 为 `0.6182`，format error rate 为
  `0.1564`，S32 单步对应 `0.7227 / 0.1836`。这些数值只描述训练遥测，不替代 CoreDev
  benchmark。No-Tool 合同全程满足：8,192 条 trajectory 中
  `successful_tgvf_observations=0`、`tool_call_attempt_rate=0.0`、
  `mean_tool_call_attempts=0.0`。S8、S16、S32 permanent checkpoint 均包含 receipt、8 份
  model、8 份 optimizer、8 份 extra-state shard 与 paired project state，各 40 个文件。
  Step 9 后 answer judge 曾出现一次 HTTP 429 transient-window failure；守护器保留已提交
  checkpoint，冷却后从同一正式 lineage 自动恢复，最终 prompts/trajectories 计数与冻结合同
  精确一致。训练成功后的 vLLM teardown 输出 `pure virtual method called`，发生在最终 metrics、
  permanent receipt 与 100% progress 之后，不影响 S32 验收。下一步只执行事前冻结的
  matched no-tool / raw-direct 双协议评测，不依据训练 reward 改选 checkpoint。

## 6. Atomic 纳入正文的决策门槛

Atomic 进入正文核心方法必须同时满足：

1. target-only Macro* 和主要 Atomic favorable sub-benchmark 不发生足以推翻当前定位的崩塌；
2. target audit 的 all-pass rate 及逐标准结果可被透明报告；
3. 文章只声称观察到的任务条件优势，不声称未被单变量消融证明的 synergy；
4. Original、Crop、TGVF 和 Atomic 四列同时出现，且 MathVerse Vision Intensive 等负面切片保留。

如果任一条件不满足，Atomic 降级到 exploratory analysis / appendix。Pure TGVF + RP67 utility
仍作为主机制线，Crop 作为强基线。

**当前决策：性能门槛部分闭合，Atomic 仍维持 exploratory。** target-only Macro* 为
`60.8253`，仍比 Original 高 `5.4697 pp`，但比自身 matched 低 `2.2574 pp`；其中 HR cross
从 `68.00%` 降至 `60.00%`，Relative Reflectance 从 `70.00%` 降至 `56.67%`，Vision Only
保持 `51.00%`。这不构成整体崩塌，却说明 matched favorable regimes 有明显衰减，尚不足以升级
为稳定核心方法。第 2 项正式 target audit 仍未完成，因此当前只保留探索性正文/附录定位。

## 7. Claim–evidence–boundary 台账

| Claim | 当前证据 | 必须保留的边界 | 状态 |
|---|---|---|---|
| TGVF 改善一组 target-conditioned reasoning regimes | Relative Depth；MathVista arithmetic、word、numeric、visual QA；逐题案例 | 不是总体最优；OCR 和精细像素读取弱；BLINK 切片小 | 可写，待 CI |
| RP67 D 具有内容 utility 和 target specificity | correct−zero `+7.15 pp`；correct−wrong `+21.57 pp`；两者 95% CI 不跨零 | diagnostic semantic overlay；oracle target 不测自主工具选择 | 已支持 |
| Atomic matched prompt 下在跨图、反射率和 Vision Only 上显示优势 | matched HR cross、BLINK reflectance、MathVerse Vision Only | target-only 下 HR cross / reflectance 优势明显收窄，Vision Only 保持；target 合格率未闭合 | 探索性，核心门槛部分验证 |
| 广义 prompt bundle 下工具总体增益仍存在 | TGVF / Atomic stress-test Macro* 为 `58.5138 / 60.4684`，分别比 Original 高 `3.1582 / 5.1128 pp` | 相对各自 matched prompt 下降 `1.2949 / 2.6142 pp`；非 target-only；不是新 benchmark 泛化 | 已支持，带退化边界 |
| 只补充 target 定义与案例的稳健性 | TGVF / Atomic target-only Macro* `58.1788 / 60.8253`，仍比 Original 高 `2.8233 / 5.4697 pp` | 相对 own matched 分别下降 `1.6298 / 2.2574 pp`；不支持“详细 target 定义普遍增益” | 已支持，带退化边界 |
| 工具方法整体优于 raw direct Original | 三个选定 checkpoint 的 Macro* 均高于 55.36 | Original 非 paired control；MathVista 等单项仍可能更强 | 可写 |
| 工具方法的增益不能只用 RL 本身解释 | No-Tool RL S32；matched no-tool 与 raw-direct 双协议；S0/S8/S16 学习动态 | 尚未出结果；工具 schema / agent-loop 差异仍存在 | 已冻结，待运行 |
| 三种方法形成不同工具调用行为 | Crop/TGVF/Atomic successful-use rate `67.05/89.73/83.17%`；calls/question `0.673/0.898/1.027`；逐 set 表 | matched-prompt 描述性统计；调用更多不等于 utility 更高；policy 自选择混杂 | 已支持 |

## 8. 论文实验部分建议结构

1. **Comprehensive comparison.** Original、No-Tool RL 与三个工具方法的七套 benchmark 主表，
   Original 永不缺席。
2. **Does RL alone explain the gain?** 冻结 No-Tool RL S32 主终点，并同时报告 matched no-tool
   与 raw-direct transfer。
3. **Where target-conditioned evidence helps.** 预冻结 sub-benchmark 图，突出 TGVF 的关系、
   深度和视觉数学优势，同时给负面切片。
4. **How do the policies use their tools?** 报告整体及逐 set 的调用覆盖率、调用强度、重复调用和
   无效尝试，并与正确率分析分开。
5. **Does RP67 carry target-specific answer utility?** 867 样本 correct/zero/wrong 三臂与
   paired effect。
6. **Robustness to target specification.** 以 target-only matched arm 为主验证；广义 full
   prompt 只作 supplementary stress test。
7. **Exploratory Atomic Crop+TGVF.** target-only 与 target audit 闭合后再决定正文或附录层级。
8. **Qualitative mechanisms and failures.** 复用已有真实 trajectory 与 bbox 案例，但把机制语言
   限定为 behavior-level inference。

## 9. 当前推进顺序

1. [已完成] RP67 semantic overlay 和 CI，机制主张已锁定；
2. [已完成] 广义 full-prompt stress test 及七项官方评分；
3. [已完成] matched-prompt 三方法整体、逐 set 调用率、调用次数和错误类型审计；
4. [已完成] TGVF S64、Atomic S16 target-only matched prompt 推理与七套官方评分；
5. [训练完成，评测待执行] No-Tool RL：合同、实现、CPU 回归、真实 canary 与正式 S32 均已
   闭合；S8/S16/S32 永久 checkpoint 已验收，下一步执行 S0/S8/S16/S32 双协议评测；
6. [待执行] 从 matched/target-only inference JSONL 物化正式 Atomic
   blind audit pack；
7. [待回填] target-only 调用行为对照与正式 audit；
8. [待写作] 形成英文 Experiments/Discussion 初稿；
9. [明确不做] Crop seed43。

## 10. 证据来源

- Original 定义和 Macro* 合同：
  `docs/POLICY_RL_COREDEV2511_MEASUREMENT_CONTRACT_AND_BASELINES_20260812.md`
- 80-step 三线路数值与 checkpoint 选择：
  `docs/PRL25_BS16_TEACHER25_80STEP_PHASE3_PLAN_20260820.md`
- 三方法逐题轨迹与失败案例：
  `docs/PRL25_CROP_TGVF_ATOMIC_QUALITATIVE_CASE_ANALYSIS_20260825.md`
- 工具调用行为：三个 matched 最佳 checkpoint 各自 `step80/step64/step16/inference/rank-0..3.jsonl`
  的 `tool_calls`、`tool_errors` 与 `successful_observation_count`；三臂均为 2,240 个唯一共同 ID。
- Crop S80 官方 summary：
  `artifacts/policy/PRL-25-B-qwen3-instruct-full-crop-exact-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-B-CROP-EXACT-COREDEV2511-STEP80-TEMP1-SEED42-UNIFIED-V1/step80/scoring/coredev-official-v1/coredev-2511-eval-summary.json`
- TGVF S64 / Atomic S16：对应 six-point evaluation 的 `step64` / `step16` 官方 summary 与
  `paired-summary.json`。

注意：主仓当前 qualitative 文档可能尚未进入本 worktree 的提交历史；本文只把它作为只读证据
来源，不覆盖主仓未提交内容。

## Appendix A. 完整 aligned sub-benchmark 清单

本附录不只保留有利切片，而是完整列出当前四种方法可稳定对齐的 sub-benchmark。除 MMMU
subject 重算外共有 40 行；其中 TGVF 或 Atomic 在 `22/40` 行上高于 Original。该计数仅用于
清点覆盖面：MathVista skill 等标签彼此重叠，不能把 `22/40` 当作独立样本上的显著性检验或
新的综合指标。小样本切片（尤其 BLINK `n=30` 和 MMMU subject `n=5–10`）只作能力定位，
不能单独承担论文核心结论。

### A.1 VStar、HRBench 与 BLINK 共同单图切片

| Sub-benchmark | n | Original | Crop S80 | TGVF S64 | Atomic S16 |
|---|---:|---:|---:|---:|---:|
| VStar / direct attributes | 115 | 48.70 | **83.48** | 70.43 | 69.57 |
| VStar / relative position | 76 | 53.95 | 78.95 | **80.26** | 75.00 |
| HRBench / cross-image aggregate | 100 | 59.00 | 61.00 | 64.00 | **68.00** |
| HRBench / single-image aggregate | 100 | 59.00 | **88.00** | 69.00 | 79.00 |
| BLINK / Counting | 30 | 66.67 | 70.00 | **73.33** | **73.33** |
| BLINK / IQ Test | 30 | **40.00** | 26.67 | 23.33 | 10.00 |
| BLINK / Object Localization | 30 | 56.67 | 56.67 | 70.00 | **73.33** |
| BLINK / Relative Depth | 30 | 83.33 | 83.33 | **86.67** | 80.00 |
| BLINK / Relative Reflectance | 30 | 50.00 | 30.00 | 46.67 | **70.00** |
| BLINK / Spatial Relation | 30 | **96.67** | 86.67 | 93.33 | 90.00 |

VStar 和 HRBench 已列出各自全部官方对齐切片。BLINK 官方 full-420 中其余 8 个类别为多图
输入，对当前工具方法不受支持并被填零；本表完整列出四种方法共同支持的 6 个单图类别。

### A.2 MathVista MINI 全部 12 个官方 Task&Skill 标签

前 5 行 task 互斥并完整覆盖 300 题；后 7 行 skill 可重叠，同一题可能进入多个 skill。

| Sub-benchmark | n | Original | Crop S80 | TGVF S64 | Atomic S16 |
|---|---:|---:|---:|---:|---:|
| MathVista task / figure question answering | 96 | **73.96** | 70.83 | 69.79 | 69.79 |
| MathVista task / geometry problem solving | 49 | **91.84** | 77.55 | 85.71 | 87.76 |
| MathVista task / math word problem | 63 | 77.78 | 68.25 | **84.13** | 79.37 |
| MathVista task / textbook question answering | 50 | **70.00** | 64.00 | 56.00 | 56.00 |
| MathVista task / visual question answering | 42 | 54.76 | 50.00 | **64.29** | 50.00 |
| MathVista skill / algebraic reasoning | 75 | **84.00** | 70.67 | 74.67 | 76.00 |
| MathVista skill / arithmetic reasoning | 104 | 65.38 | 57.69 | **72.12** | 63.46 |
| MathVista skill / geometry reasoning | 63 | **87.30** | 73.02 | 79.37 | 79.37 |
| MathVista skill / logical reasoning | 12 | **41.67** | 33.33 | 16.67 | 25.00 |
| MathVista skill / numeric commonsense | 36 | 47.22 | 44.44 | **58.33** | 50.00 |
| MathVista skill / scientific reasoning | 37 | **62.16** | **62.16** | 54.05 | 56.76 |
| MathVista skill / statistical reasoning | 111 | **82.88** | 80.18 | 81.98 | 81.08 |

四种方法均使用完整的相同 300 个 `index`。作为互斥的附加诊断，free-form / multi-choice 的
样本数为 `171 / 129`：Original 为 `64.91 / 86.82`，Crop S80 为 `59.65 / 77.52`，TGVF
S64 为 `65.50 / 81.40`，Atomic S16 为 `61.40 / 80.62`。这定位出 TGVF 总体回退主要落在
multi-choice，而不是评测 subset 改变。

### A.3 MathVerse MINI 全部 5 个版本

| Sub-benchmark | n | Original | Crop S80 | TGVF S64 | Atomic S16 |
|---|---:|---:|---:|---:|---:|
| MathVerse / Text Dominant | 100 | **69.00** | 64.00 | 64.00 | 66.00 |
| MathVerse / Text Lite | 100 | 53.00 | 56.00 | 58.00 | **59.00** |
| MathVerse / Vision Dominant | 100 | **51.00** | 45.00 | 46.00 | 50.00 |
| MathVerse / Vision Intensive | 100 | **52.00** | 49.00 | 42.00 | 49.00 |
| MathVerse / Vision Only | 100 | 28.00 | 43.00 | 42.00 | **51.00** |

### A.4 OCRBench v2 全部 13 个官方语言类别

| Sub-benchmark | n | Original | Crop S80 | TGVF S64 | Atomic S16 |
|---|---:|---:|---:|---:|---:|
| OCR EN / text recognition | category | 60.49 | 70.23 | 55.05 | **73.38** |
| OCR EN / text detection | category | 29.00 | 24.55 | 28.87 | **36.05** |
| OCR EN / text spotting | category | 0.00 | 10.00 | 16.50 | **18.90** |
| OCR EN / relationship extraction | category | 89.05 | **90.55** | 76.63 | 88.14 |
| OCR EN / element parsing | category | **43.89** | 39.67 | 29.27 | 40.25 |
| OCR EN / mathematical calculation | category | 39.25 | **43.67** | 34.14 | 32.82 |
| OCR EN / visual text understanding | category | 75.00 | 81.61 | 80.00 | **83.21** |
| OCR EN / knowledge reasoning | category | **62.44** | 56.35 | 57.50 | 57.47 |
| OCR CN / text recognition | category | 59.82 | **71.23** | 22.17 | 67.80 |
| OCR CN / relationship extraction | category | 49.31 | **77.13** | 55.82 | 58.81 |
| OCR CN / element parsing | category | 27.75 | 28.94 | 20.69 | **33.14** |
| OCR CN / visual text understanding | category | 35.00 | 60.00 | **65.00** | 55.00 |
| OCR CN / knowledge reasoning | category | **60.51** | 55.68 | 45.55 | 59.07 |

OCR 类别分数来自官方 rule-based scorer；各类别内部样本数和计分尺度不同，因此不对这 13 行
再做无权平均。

### A.5 MMMU-Pro 共同 269 单图的完整 subject 审计

MMMU 官方 subject 表混有工具方法不支持的 31 个多图样本，不能直接与 Original 比。下表以
三个工具 result TSV 的 `extra_records.coverage` 共同支持标记确定 269 个单图 ID，再与 Original
result TSV 按相同 `index` 对齐并按 subject 聚合 `hit`。这是由官方逐题判分导出的 aligned
diagnostic，不是 MMMU 官方 subject headline。

| Sub-benchmark | n | Original | Crop S80 | TGVF S64 | Atomic S16 |
|---|---:|---:|---:|---:|---:|
| MMMU / Accounting | 10 | 20.00 | 50.00 | **60.00** | **60.00** |
| MMMU / Agriculture | 10 | **40.00** | 30.00 | **40.00** | 30.00 |
| MMMU / Architecture and Engineering | 10 | 40.00 | 50.00 | 50.00 | **60.00** |
| MMMU / Art | 10 | **50.00** | 30.00 | 30.00 | 30.00 |
| MMMU / Art Theory | 7 | **57.14** | **57.14** | **57.14** | **57.14** |
| MMMU / Basic Medical Science | 10 | 30.00 | 10.00 | 20.00 | **50.00** |
| MMMU / Biology | 7 | **42.86** | 14.29 | 0.00 | 14.29 |
| MMMU / Chemistry | 5 | 20.00 | 40.00 | **60.00** | **60.00** |
| MMMU / Clinical Medicine | 9 | 11.11 | 11.11 | **22.22** | **22.22** |
| MMMU / Computer Science | 10 | **70.00** | **70.00** | 40.00 | 50.00 |
| MMMU / Design | 10 | 70.00 | **80.00** | 70.00 | 60.00 |
| MMMU / Diagnostics and Laboratory Medicine | 10 | **20.00** | 10.00 | 10.00 | **20.00** |
| MMMU / Economics | 8 | **50.00** | 37.50 | 25.00 | 37.50 |
| MMMU / Electronics | 10 | 40.00 | 60.00 | 50.00 | **70.00** |
| MMMU / Energy and Power | 10 | **40.00** | 20.00 | 30.00 | 30.00 |
| MMMU / Finance | 10 | 10.00 | 60.00 | **70.00** | **70.00** |
| MMMU / Geography | 8 | **50.00** | 37.50 | 25.00 | 37.50 |
| MMMU / History | 8 | 50.00 | **62.50** | **62.50** | **62.50** |
| MMMU / Literature | 8 | 75.00 | **87.50** | 75.00 | 75.00 |
| MMMU / Manage | 10 | **50.00** | 40.00 | 40.00 | **50.00** |
| MMMU / Marketing | 9 | 33.33 | 66.67 | 88.89 | **100.00** |
| MMMU / Materials | 8 | 12.50 | 25.00 | 50.00 | **75.00** |
| MMMU / Math | 10 | **60.00** | 30.00 | 40.00 | 40.00 |
| MMMU / Mechanical Engineering | 10 | 0.00 | **50.00** | 40.00 | 40.00 |
| MMMU / Music | 7 | 42.86 | **71.43** | 42.86 | 42.86 |
| MMMU / Pharmacy | 7 | 14.29 | 42.86 | 57.14 | **71.43** |
| MMMU / Physics | 10 | 30.00 | 50.00 | 40.00 | **60.00** |
| MMMU / Psychology | 8 | 50.00 | **62.50** | 25.00 | **62.50** |
| MMMU / Public Health | 10 | 40.00 | **90.00** | 80.00 | 60.00 |
| MMMU / Sociology | 10 | **50.00** | **50.00** | **50.00** | **50.00** |

TGVF 或 Atomic 在 `15/30` 个 subject 上高于 Original；其中 Marketing、Materials、Finance、
Pharmacy、Accounting 的描述性增益最大。由于每个 subject 仅 `n=5–10`，这些行只用于提出
能力假设和挑选案例，不进入 Macro*。
