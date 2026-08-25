# PRL25 Crop、TGVF 与 Atomic Crop+TGVF 定性案例分析

日期：2026-08-25（Asia/Tokyo）

用途：聚合论文实验部分所需的三类方法特性、逐题分叉案例、失败模式和机制解释边界。

## 1. 核心结论

在相同的 CoreDev-2511 支持集和 seed42 评测采样下，native Crop、pure TGVF 与
Atomic Crop+TGVF 呈现出互补而非简单包含的能力结构：Crop 更擅长保留局部原始像素，
pure TGVF 更擅长按语义寻找目标并整合全局关系，Atomic Crop+TGVF 更擅长在空间定位后继续
进行比较或计算。三者也分别受到裁剪失位、latent 精确信息损失和过度分解的限制。

这一结论由总体分数、逐样本胜负、工具轨迹和原图核验共同支持，但仍是行为层面的机制推断。
当前结果不能单独证明某个内部表征变化是这些差异的唯一原因。

## 2. 术语与比较对象

| 本文简称 | 固定含义 | 工具与返回观察 |
|---|---|---|
| **Crop** | PRL25-B native Crop，S32 | `image_zoom_in_tool`；返回原图上的 RGB crop |
| **TGVF** | PRL25-C pure TGVF，S64 | `tgvf_focus_tool`；由语义 `target` 请求 Frozen RP67 latent observation |
| **Atomic** | PRL25-D Atomic Crop+TGVF，S16 | `tgvf_crop_tool`；同时提交 `bbox_2d` 与语义 `target`，返回 crop-conditioned latent observation |
| **Macro\*** | 七项等权聚合指标 | VStar、HRBench、BLINK、OCR EN/CN mean、MMMU、MathVista、MathVerse |

三条线路均使用 Qwen3-VL-8B-Instruct、BS16 × n16、Teacher25、T-free reward、FMT2
错误罚分 `-2` 和 80-step fresh-start 训练合同。三条线路的工具和 prompt schema 不同，
因此横向差异用于建立能力地图，不能称为严格的单变量因果消融或 synergy。

## 3. 对齐评测口径

案例分析固定使用三条线路共同的 seed42 结果。每条线路均覆盖相同的 2,240 个受支持样本：

| 数据集 | 对齐样本数 |
|---|---:|
| VStar | 191 |
| HRBench | 200 |
| BLINK | 180 |
| OCRBench v2 | 600 |
| MMMU | 269 |
| MathVista | 300 |
| MathVerse | 500 |
| **合计** | **2,240** |

TGVF S64 和 Atomic S16 另有 seed43 独立复测。seed43 只用于检查所选 checkpoint 的聚合
稳定性，不用于三方法逐题案例比较；Crop S32 当前没有对应的 seed43 复测。

| 方法 | 选定 checkpoint | Seed42 Macro* | Seed43 Macro* | 双-seed 均值 | 最佳观测值 |
|---|---:|---:|---:|---:|---:|
| Crop | S32 | **63.5377** | — | — | **63.5377** |
| TGVF | S64 | 59.8086 | **60.1807** | 59.9947 | **60.1807** |
| Atomic | S16 | **63.0827** | 62.8952 | 62.9889 | **63.0827** |

TGVF 和 Atomic 的 Macro* 跨 seed 绝对变化分别为 `0.3721 pp` 和 `0.1875 pp`。这支持
聚合结果在两次采样间较稳定，但两个 seed 不足以估计正式方差或置信区间；单项数据集仍可
波动约 `1--4 pp`。

## 4. 总体能力分布

下表全部使用共同 seed42，单位为 `%`。

| 指标 | Crop S32 | TGVF S64 | Atomic S16 |
|---|---:|---:|---:|
| VStar | **80.1047** | 74.3455 | 71.7277 |
| HRBench | 73.0000 | 66.5000 | **73.5000** |
| BLINK single-image | 64.4444 | 65.5556 | **66.1111** |
| OCR EN/CN mean | **54.8108** | 44.5446 | 54.2720 |
| MMMU single-image | 49.0706 | 44.9814 | **51.3011** |
| MathVista | 71.3333 | **72.3333** | 69.6667 |
| MathVerse macro | 52.0000 | 50.4000 | **55.0000** |
| **Macro\*** | **63.5377** | 59.8086 | 63.0827 |

总体结果首先显示，Crop 与 Atomic 的聚合分数接近，但优势来源不同。Crop 的领先主要来自
VStar 和 OCR；Atomic 在 HRBench、BLINK、MMMU 与 MathVerse 更强。TGVF 的总体分数较低，
但在 MathVista 和一部分语义定位、深度及跨区域关系题上仍有不可被另外两种方法替代的正确
案例。

## 5. 逐题互补性

OCR 的官方评分不是简单二值题，因此本节只统计其余 1,640 个样本。三方法逐题结果模式如下：

| Crop / TGVF / Atomic | 样本数 | 含义 |
|---|---:|---|
| 1 / 1 / 1 | 770 | 三者都正确 |
| 1 / 1 / 0 | 67 | Crop 与 TGVF 正确 |
| 1 / 0 / 1 | 92 | Crop 与 Atomic 正确 |
| 0 / 1 / 1 | 96 | TGVF 与 Atomic 正确 |
| 1 / 0 / 0 | 92 | 仅 Crop 正确 |
| 0 / 1 / 0 | 50 | 仅 TGVF 正确 |
| 0 / 0 / 1 | 67 | 仅 Atomic 正确 |
| 0 / 0 / 0 | 406 | 三者都错误 |

Crop 与 TGVF 的直接分叉为 Crop 胜 `184`、TGVF 胜 `146`；TGVF 与 Atomic 为 TGVF 胜
`117`、Atomic 胜 `159`；Crop 与 Atomic 则为 Crop 胜 `159`、Atomic 胜 `163`。最后一组
几乎平齐，但双方各有大量独占正确样本，说明相近的 Macro* 掩盖了明显不同的能力组成。

| 数据集 | 仅 Crop 正确 | 仅 TGVF 正确 | 仅 Atomic 正确 |
|---|---:|---:|---:|
| VStar | **17** | 5 | 3 |
| HRBench | **16** | 7 | 8 |
| BLINK | **11** | 7 | 6 |
| MMMU | 16 | 8 | **20** |
| MathVista | 11 | **11** | 5 |
| MathVerse | 21 | 12 | **25** |

## 6. 三类方法的行为特性

### 6.1 Crop：局部像素保真与选择性调用

Crop 在 2,240 个样本中的工具使用率为 `47.63%`。当策略判断原图足以回答时，它更常保留
直接回答路径；需要细节时则返回真实 RGB crop。这一行为与其 VStar 和 OCR 优势一致。

代表性成功案例为 `HRBench4K/0159_0.jpg`。问题询问教堂日晷上刻写的年份，正确答案为
`1762`。年份只占 4K 原图中的极小区域：

- Crop 使用 bbox `[118, 664, 226, 737]` 放大日晷，回答 `1762`；
- TGVF 以“日晷铭文”为 target，回答 `1782`；
- Atomic 使用更窄的 bbox，回答 `1768`。

该案例直接支持 Crop 对小字和细粒度像素的优势。另一个案例
`VStarBench/0046_0.jpg` 要判断人物帽子颜色，Crop 放大极小区域后正确回答白色，而 TGVF
与 Atomic 均回答黑色。

Crop 的主要失效模式是空间选择错误和运行级开销。它出现 42 个多次调用样本、71 个带工具
错误的轨迹和 8 个 context-limit 停止。`HRBench4K/0075_0.jpg` 中，Crop 因达到上下文上限
没有形成最终答案，而 TGVF 与 Atomic 均答对。局部 crop 也可能切断跨区域关系，使模型在
只看到一处细节时无法恢复全局比较。

### 6.2 TGVF：语义定位与全局关系

TGVF 的工具使用率为 `89.73%`，说明策略几乎把语义 focus 当作默认观察路径。它不要求模型
先给出精确 bbox，因此在目标隐蔽、目标名称明确或需要整体关系时具有优势。

`HRBench4K/0178_0.jpg` 要判断地图上哪些编号位置属于同一个国家，正确答案为位置 1 和 4：

- TGVF 将四个位置作为一个统一语义目标观察，回答 1 和 4；
- Crop 只放大位置 4，回答错误；
- Atomic 顺序查看局部区域，但未恢复正确的全局关系。

同类案例包括：

- `VStarBench/0004_0.jpg`：TGVF 正确识别背黄色背包的女性正在蹲下，Crop 与 Atomic
  均判断为站立；
- `HRBench4K/0093_0.jpg`：TGVF 通过语义 target 找到隐蔽的监控摄像头并判断其颜色，
  另外两种方法未找到该目标；
- `BLINK/0064_0.jpg`：TGVF 正确处理相对深度关系，Crop 与 Atomic 错误。

TGVF 的短板集中在精确读取与输出控制。它有 42 个 max-token 样本，远高于正常的单次工具
调用数量所能解释的范围。`VStarBench/0009_0.jpg` 中，纸巾盒的正确颜色为蓝色；Crop
回答正确，而 TGVF 将其识别为棕红色并重复生成至 20,480 tokens。该现象同时影响 OCR：
TGVF 有时能读取局部内容，却将答案扩写成摘要或多次重复，而非按评测要求精确转录。

### 6.3 Atomic：定位、观察与后续推理的组合

Atomic 的工具使用率为 `83.17%`，平均每个样本 `1.027` 次调用；221 个样本发生多次调用。
它同时提供 bbox 和语义 target，使策略能够把“看哪里”和“从该区域提取什么”组合起来，
并在观察后继续执行比较或计算。

`MathVerse_MINI/0206_0.png` 给出相对北方偏西 `31°` 的射线，并询问方位角。正确答案为
`329°`：

- Crop 和 TGVF 均直接返回图中的 `31°`；
- Atomic 定位方向和角度后完成 `360-31=329`，回答正确。

`BLINK/0303_0.jpg` 要比较 A、B 两点的反射率。Atomic 分别观察两个局部区域并正确判断
B 更暗；Crop 判断相同，TGVF 判断 A 更暗。`MathVerse_MINI/0178_0.png` 中，Atomic 也能
通过定位椭圆范围计算出正确中心 `(20, 10)`，另外两种方法分别回答 `(0, 0)` 和 `(10, 15)`。

Atomic 的主要风险是过度分解。`BLINK/0144_0.jpg` 是需要同时观察整个九宫格规律的 IQ
题，正确选项为 C。Atomic 连续进行 6 次局部调用，触发工具调用上限并答错；Crop 保留整体
结构后答对。Atomic 总计出现 21 次 `tool_call_limit_exceeded`。这些案例表明，多步局部观察
并不自动等价于更强的全局组合能力。

## 7. OCR 进一步揭示的信息保真差异

OCRBench v2 的官方分项结果显示，Crop 与 Atomic 显著优于 pure TGVF：

| OCR 分项 | Crop | TGVF | Atomic |
|---|---:|---:|---:|
| English overall | 52.22 | 47.24 | **53.78** |
| Chinese overall | **57.40** | 41.84 | 54.77 |
| English text recognition | 68.76 | 55.05 | **73.38** |
| Chinese text recognition | **73.99** | 22.17 | 67.80 |
| English relationship extraction | **88.18** | 76.63 | 88.14 |
| Chinese relationship extraction | **74.31** | 55.82 | 58.81 |

`OCRBench_v2/0373_0.jpg` 是一页包含多栏中文正文的杂志页面。Crop 与 Atomic 都返回了接近
完整的逐字转录；TGVF 则将长文本压缩为不完整内容并发生重复。类似模式也出现在中文票据
`OCRBench_v2/0226_0.jpg` 和英文收据 `OCRBench_v2/0557_0.jpg`：Crop 与 Atomic 更容易
给出简洁的结构化字段，TGVF 更容易重复已经读取到的内容。

这些观察与“native pixels 更有利于精确 OCR、latent observation 更偏语义信息”的解释
一致，但不能仅凭当前结果把差异完全归因于 RP67 压缩。工具使用率、policy 生成习惯、停止
行为和训练数据分布也可能共同造成该结果。

## 8. 机制解释与证据边界

| 行为结论 | 直接证据 | 当前解释 | 状态 |
|---|---|---|---|
| Crop 更适合小字、颜色和局部属性 | VStar、OCR 聚合优势；日晷与帽子案例 | RGB crop 保留局部像素带宽 | **支持行为结论；机制为推断** |
| TGVF 更适合语义定位与部分全局关系 | 地图、摄像头、姿态和深度案例 | 语言 target 减少了显式 bbox 定位负担 | **支持行为结论；机制为推断** |
| Atomic 更适合定位后的比较与计算 | 方位角、椭圆中心、反射率案例 | bbox 与 target 联合约束观察，并触发后续推理 | **支持行为结论；机制为推断** |
| TGVF 的精确 OCR 较弱 | 官方 OCR 分项与重复生成轨迹 | latent 精确信息损失和 policy 输出退化共同作用 | **现象已支持；原因未分离** |
| Atomic 可能因过度分解损失全局结构 | IQ Test 多次调用和 tool-call-limit 案例 | 多个局部 observation 难以还原整体规律 | **现象已支持；原因未分离** |

要把上述机制推断提升为因果结论，至少需要以下控制实验：

1. 在相同 policy 与问题上强制 direct、Crop、TGVF 和 Atomic，隔离工具选择偏差；
2. 对同一 bbox 比较 native RGB crop 与 crop-conditioned latent，隔离 observation 表征；
3. 固定一次调用与允许多次调用，隔离 Atomic 的分解策略；
4. 对 OCR 使用相同最大输出长度和停止约束，区分感知错误与重复生成错误。

因此，工具调用样本与直接回答样本之间的平均分差只能描述策略选择，不能解释为调用工具的
因果收益；三条线路的最佳 checkpoint 也是根据各自 seed42 曲线 post-hoc 选出，案例分析
描述的是所选 operating point，不替代预注册 S80 主终点。

## 9. 建议用于论文的案例图

建议将定性图组织为三列，每列包含一个主要成功案例和一个边界案例：

| 列 | 成功案例 | 边界或失败案例 | 要表达的信息 |
|---|---|---|---|
| Crop | `HRBench4K/0159_0.jpg` 日晷年份 | `HRBench4K/0075_0.jpg` context limit | 像素保真强，但裁剪和序列开销可能失败 |
| TGVF | `HRBench4K/0178_0.jpg` 地图跨区域关系 | `VStarBench/0009_0.jpg` 颜色误判与重复 | 语义定位强，但精确读取和输出控制较弱 |
| Atomic | `MathVerse_MINI/0206_0.png` 方位角 | `BLINK/0144_0.jpg` 九宫格过度分解 | 定位后推理强，但局部调用可能破坏全局结构 |

可增加一行共享 OCR 案例 `OCRBench_v2/0373_0.jpg`，并列展示三者完整输出。该案例能直观
连接 qualitative trace 与 OCR 分项统计，而不是只展示偶然的选择题胜负。

## 10. 可安全使用的论文表述

可以安全使用：

- “The three visual interaction mechanisms exhibited complementary capability profiles under an aligned evaluation set.”
- “Native cropping was strongest for pixel-sensitive local evidence, whereas semantic TGVF queries recovered targets and relations that were missed by coordinate-only localization.”
- “Atomic Crop+TGVF was effective when localized evidence required a subsequent comparison or computation, but repeated local observations could impair tasks that depended on global structure.”

需要避免：

- “Atomic 严格结合了 Crop 与 TGVF 的全部优势”：它仍有独立失败区域；
- “Frozen RP67 导致全部 OCR 退化”：当前没有隔离 policy 和 observation 的 matched ablation；
- “调用工具使准确率提高了多少”：工具使用由题目难度和 policy 共同选择，存在选择偏差；
- “S32/S64/S16 是预注册终点”：它们是各自曲线中的 post-hoc 最优 checkpoint。

## 11. Artifact 来源

Crop S32：

```text
artifacts/policy/PRL-25-B-qwen3-instruct-full-crop-exact-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-B-CROP-EXACT-COREDEV2511-STEP8-STEP32-STEP48-STEP64-TEMP1-SEED42-UNIFIED-V1/step32/
```

TGVF S64 seed42：

```text
artifacts/policy/PRL-25-C-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-teacher25-80step-ws8/evaluation/PRL25-C-FROZEN-RP67-TFREE-TEACHER25-COREDEV2511-S8-S16-S32-S48-S64-S80-PAIRED-SEED-V1/step64/
```

Atomic S16 seed42：

```text
artifacts/policy/PRL-25-D-qwen3-instruct-full-frozen-rp67-bs16-n16-tfree-crop-tgvf-teacher25-80step-ws8/evaluation/PRL25-D-ATOMIC-CROP-TGVF-RP67-TFREE-TEACHER25-COREDEV2511-S8-S16-S32-S48-S64-S80-PAIRED-SEED-V1/step16/
```

相关总表与训练合同：

- [PRL25 第三期 BS16 Teacher25 80-step 计划](PRL25_BS16_TEACHER25_80STEP_PHASE3_PLAN_20260820.md)
- [BS16 Crop、TGVF 与 Crop+TGVF 历史对比资料页](BS16_CROP_TGVF_REWARD_ALIGNED_ANALYSIS_20260820.md)
- [实验账本](EXPERIMENT_LEDGER.md)
