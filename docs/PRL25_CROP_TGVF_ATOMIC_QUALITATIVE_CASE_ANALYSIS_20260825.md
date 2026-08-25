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

## 6. 三类方法的行为特性与扩充案例

本节中的 bbox 均写作 `[x1, y1, x2, y2]`，坐标空间为 Qwen3 相对坐标 `0--1000`。
`direct` 表示该方法没有调用视觉工具。每个案例均列出三条 seed42 轨迹中的真实 Crop
`bbox + label`、TGVF `target`、Atomic `bbox + target`，而不是根据最终答案反推模型看了
哪里。高分辨率图片的展示预览可能在工具 bbox 外保留少量上下文；精确请求范围以表格为准。

### 6.1 Crop：局部像素保真与选择性调用

Crop 在 2,240 个样本中的工具使用率为 `47.63%`。当策略判断原图足以回答时，它更常保留
直接回答路径；需要细节时则返回真实 RGB crop。这一行为与其 VStar 和 OCR 优势一致。

| C1：日晷小字 | C2：远处帽子颜色 |
|---|---|
| <img src="assets/prl25_qualitative_cases/crop_sundial_hr0159.jpg" width="420"><br>`HRBench4K/0159_0.jpg` | <img src="assets/prl25_qualitative_cases/crop_cap_vstar0046.jpg" width="320"><br>`VStarBench/0046_0.jpg` |
| C3：保留全局左右关系 | C4：Crop context-limit 失败 |
| <img src="assets/prl25_qualitative_cases/crop_chair_vstar0190.jpg" width="420"><br>`VStarBench/0190_0.jpg` | <img src="assets/prl25_qualitative_cases/crop_bridge_hr0075.jpg" width="420"><br>`HRBench4K/0075_0.jpg` |

| 案例与正确答案 | Crop 轨迹与回答 | TGVF 轨迹与回答 | Atomic 轨迹与回答 |
|---|---|---|---|
| **C1 日晷年份：1762** | `bbox=[118,664,226,737]`，`label="a close-up of the sundial on the building"`；对应原图约 `[475,2677,911,2971] px`；答 **1762 ✓** | `target="the inscription on the sundial on the left side of the building"`；答 1782 ✗ | `bbox=[119,690,171,743]`，`target="a close-up of the sundial to read the inscribed year"`；答 1768 ✗ |
| **C2 帽子颜色：white** | `bbox=[10,589,25,644]`，`label="a man wearing a cap"`；对应原图约 `[22,883,56,966] px`；答 **white ✓** | `target="the man in the background wearing a cap and his cap's color"`；答 black ✗ | `bbox=[11,586,33,651]`，`target="a man wearing a cap, to identify the color of the cap"`；答 black ✗ |
| **C3 白椅位于街道右侧** | **direct**；答 **right ✓** | `target="the white chairs located along the street"`；答 left ✗ | `bbox=[333,744,455,815]`，`target="the white chairs and their position relative to the street"`；答 left ✗ |
| **C4 前景儿童跑离相机** | `bbox=[292,478,705,727]`，`label="the child in the foreground running on the bridge"`；随后 `context_limit`，无最终答案 ✗ | `target="the child in the foreground on the bridge and their running direction relative to the camera"`；答 **away ✓** | `bbox=[639,497,753,743]`，`target="the child in the foreground on the bridge and their orientation relative to the camera"`；答 **away ✓** |

C1 和 C2 直接支持 Crop 对极小文字与颜色细节的优势。C3 则说明 Crop 的优势不只来自调用
工具：该策略选择 direct 后保留了完整街景和左右关系，另外两种方法在聚焦“白椅”后反而把
参考方向判断反了。C4 是重要反例，Crop 已选中正确的大区域，却因序列达到 context limit
没有形成答案。因此，局部像素质量和端到端轨迹可靠性需要分开讨论。

Crop 共出现 42 个多次调用样本、71 个带工具错误的轨迹和 8 个 context-limit 停止。它的
主要风险不是 RGB crop 本身缺少细节，而是 bbox 选择、裁剪协议失败以及局部视图切断全局
关系。

### 6.2 TGVF：语义定位与全局关系

TGVF 的工具使用率为 `89.73%`，说明策略几乎把语义 focus 当作默认观察路径。它不要求模型
先给出精确 bbox，因此在目标名称明确、姿态判别或需要统一处理多个区域时具有优势。

| T1：姿态识别 | T2：跨区域地图关系 |
|---|---|
| <img src="assets/prl25_qualitative_cases/tgvf_pose_vstar0004.jpg" width="360"><br>`VStarBench/0004_0.jpg` | <img src="assets/prl25_qualitative_cases/tgvf_map_hr0178.jpg" width="480"><br>`HRBench4K/0178_0.jpg` |
| T3：相对深度 | T4：物体空间关系 |
| <img src="assets/prl25_qualitative_cases/tgvf_depth_blink0064.jpg" width="360"><br>`BLINK/0064_0.jpg` | <img src="assets/prl25_qualitative_cases/tgvf_relation_blink0123.jpg" width="360"><br>`BLINK/0123_0.jpg` |
| T5：精确颜色与重复生成失败 |  |
| <img src="assets/prl25_qualitative_cases/tgvf_tissue_vstar0009.jpg" width="480"><br>`VStarBench/0009_0.jpg` |  |

| 案例与正确答案 | Crop 轨迹与回答 | TGVF 轨迹与回答 | Atomic 轨迹与回答 |
|---|---|---|---|
| **T1 黄色背包女性：squatting** | `bbox=[601,149,652,299]`，`label="woman with yellow backpack"`；答 standing ✗ | `target="the woman wearing a yellow backpack and her body posture"`；答 **squatting ✓** | `bbox=[615,150,680,293]`，`target="the woman with a yellow backpack and her body posture"`；答 standing ✗ |
| **T2 同属中国：位置 1 和 4** | `bbox=[696,48,863,181]`，`label="Location 4 on a map"`；只检查局部位置 4，答 1 和 2 ✗ | `target="the geographical locations marked by numbers 1, 2, 3, and 4 on the map and their respective countries"`；答 **1 和 4 ✓** | 依次调用 `bbox=[100,540,190,658]`，`target="a red circle labeled '1' on a map, indicating a geographical location"`；`[70,718,160,836]`，`target="a red circle labeled '2' on a map indicating a geographical location"`；`[850,252,950,370]`，`target="a red circle labeled '3' on a map indicating a geographical location"`；`[820,107,920,195]`，`target="a red circle labeled '4' on a map indicating a geographical location"`；答 1 和 2 ✗ |
| **T3 更靠近相机：B** | `bbox=[480,31,562,194]`，`label="point B"`；答 A ✗ | `target="the relative positions of points A and B in the image, specifically which one appears closer to the camera"`；答 **B ✓** | `bbox=[184,28,334,218]`，`target="a close-up view of point A and its immediate surroundings to assess its distance from the camera"`；`bbox=[578,28,731,218]`，`target="a close-up view of point B and its immediate surroundings to assess its distance from the camera"`；答 A ✗ |
| **T4 椅子不在马下方** | **direct**；答 yes ✗ | `target="the position of the chair relative to the horse"`；答 **no ✓** | `bbox=[43,516,325,702]`，`target="a chair positioned under the horse"`；答 yes ✗ |
| **T5 纸巾盒颜色：blue** | **direct**；答 **blue ✓** | `target="the tissue box on the truck bed and its color"`；把棕红色货箱当作目标，答 brownish-red，并重复到 `20,480` tokens ✗ | `bbox=[156,697,265,765]`，`target="a tissue box and its color"`；答 light gray ✗ |

T1 表明相同空间范围并不保证相同观察：Crop 和 Atomic 给出的框与人物位置接近，但只有
语义 focus 得到正确姿态。T2 更能区分方法结构。TGVF 用一个 target 同时绑定四个位置，
而 Crop 只看位置 4，Atomic 虽逐个查看四处，仍未恢复正确的国家级关系。T3 与 T4 提供了
两个不同数据集上的关系判断证据。

T5 则揭示 TGVF 的精确读取与输出控制问题。模型把 target 中的“truck bed”绑定到棕红色
货箱，而没有选择驾驶室挡风玻璃后的蓝色纸巾盒，并在错误判断后重复生成至上限。pure TGVF
共有 42 个 max-token 样本，因此该案例不是完全孤立的格式现象。

### 6.3 Atomic：定位、观察与后续推理的组合

Atomic 的工具使用率为 `83.17%`，平均每个样本 `1.027` 次调用；221 个样本发生多次调用。
它同时提供 bbox 和语义 target，使策略能够把“看哪里”和“从该区域提取什么”组合起来，
并在观察后继续执行比较或计算。

| A1：方位角换算 | A2：椭圆中心 |
|---|---|
| <img src="assets/prl25_qualitative_cases/atomic_bearing_mathverse0206.jpg" width="300"><br>`MathVerse_MINI/0206_0.png` | <img src="assets/prl25_qualitative_cases/atomic_ellipse_mathverse0178.png" width="360"><br>`MathVerse_MINI/0178_0.png` |
| A3：双区域反射率比较 | A4：计数四只狗 |
| <img src="assets/prl25_qualitative_cases/atomic_reflectance_blink0303.jpg" width="420"><br>`BLINK/0303_0.jpg` | <img src="assets/prl25_qualitative_cases/atomic_count_hr0114.jpg" width="420"><br>`HRBench4K/0114_0.jpg` |
| A5：扇形周长 | A6：三角函数周期 |
| <img src="assets/prl25_qualitative_cases/atomic_sector_mathverse0426.png" width="360"><br>`MathVerse_MINI/0426_0.png` | <img src="assets/prl25_qualitative_cases/atomic_period_mathverse0204.png" width="300"><br>`MathVerse_MINI/0204_0.png` |
| A7：九宫格过度分解失败 |  |
| <img src="assets/prl25_qualitative_cases/atomic_iq_blink0144.jpg" width="300"><br>`BLINK/0144_0.jpg` |  |

| 案例与正确答案 | Crop 轨迹与回答 | TGVF 轨迹与回答 | Atomic 轨迹与回答 |
|---|---|---|---|
| **A1 true bearing：329°** | **direct**；只返回图中标注的 31° ✗ | `target="the angle labeled in the diagram between the north direction and the line segment OA"`；返回 31° ✗ | `bbox=[100,100,875,838]`，`target="the diagram showing point A and the angle from the north line to the line OA, which is labeled as 31 degrees"`；识别西偏北并计算 `360-31`，答 **329° ✓** |
| **A2 椭圆中心：(20,10)** | **direct**；答 `(0,0)` ✗ | `target="the center of the ellipse in the coordinate system"`；答 `(10,15)` ✗ | `bbox=[137,533,866,787]`，`target="the center of the ellipse (cake) and its coordinates (a, b)"`；读取水平/垂直范围后答 **`(20,10)` ✓** |
| **A3 surface color：B darker** | `bbox=[337,366,400,515]`，`label="point B and its surrounding area"`；只查看 B，答 same ✗ | `target="the color of the surface at point A and the color of the surface at point B"`；答 A darker ✗ | `bbox=[268,404,319,486]`，`target="a close-up of point A to determine its surface color"`；`bbox=[392,430,436,505]`，`target="a close-up of point B to determine its surface color"`；答 **B darker ✓** |
| **A4 被牵的狗：4** | **direct**；答 3 ✗ | `target="the number of dogs being walked by the person in the image"`；答 3 ✗ | `bbox=[437,835,679,901]`，`target="the dogs being walked by the person in the red jacket"`；答 **4 ✓** |
| **A5 扇形周长：18.89** | `bbox=[132,134,354,712]`，`label="the sector with labeled radius and angle"`；答 18.94 ✗ | `target="the sector with its radius and central angle labeled"`；答 18.65 ✗ | `bbox=[120,180,880,780]`，`target="the sector with a radius of 7 cm and a central angle of 40 degrees, to calculate its perimeter"`；答 **18.89 ✓** |
| **A6 三角函数周期：π** | **direct**；误读峰值为 1 和 5，答 4 ✗ | `target="the distance between two consecutive peaks on the graph to determine the period"`；答 `2π` ✗ | `bbox=[1,1,996,991]`，`target="the graph of the trigonometric function to identify two consecutive peaks and calculate the horizontal distance between them as the period"`；答 **`π` ✓** |
| **A7 IQ Test：选项 C** | `bbox=[10,706,992,996]`，`label="bottom row of the 3x3 grid"`；答 **C ✓** | `target="the pattern of shapes in the 3x3 grid, specifically focusing on the inner shapes of the outer shapes in each cell"`；答 B ✗ | 连续调用 `[100,100,300,300]`、`[325,100,525,300]`、`[650,100,850,300]`、`[100,325,300,525]`、`[325,325,525,525]`、`[650,325,850,525]`；target 依次为 `"the first/second/third column of shapes to identify the pattern based on shape transformation"`，随后对第二行重复 first/second/third；触发 `tool_call_limit_exceeded` 并以 `max_tokens` 结束 ✗ |

A1、A2、A5 和 A6 的共同点不是单纯“看清数字”，而是从定位后的视觉证据继续完成坐标、
角度或公式推理。A3 与 A4 则分别展示双区域比较和密集小目标计数。A7 给出相反边界：
九宫格规律要求保留整体结构，Atomic 把图拆成六个局部后仍未完成组合，并触发调用上限。

Atomic 总计出现 21 次 `tool_call_limit_exceeded`。因此，多步局部观察可以帮助分解任务，
但不自动等价于更强的全局组合能力。

### 6.4 共享 OCR 案例：同一页面的三种信息路径

<img src="assets/prl25_qualitative_cases/shared_ocr0373.png" width="620">

图：`OCRBench_v2/0373_0.jpg`，任务为读取整页中文文本。

| 方法 | 精确工具请求 | 输出表现 |
|---|---|---|
| Crop | `bbox=[56,51,945,870]`，`label="the main body of the article containing text"` | 返回约 1,533 字符的连续正文，接近完整逐字转录 |
| TGVF | `target="all the text content in the image"` | 返回约 751 字符，主要是关键词、摘要式内容和重复片段，未完成逐字转录 |
| Atomic | `bbox=[69,52,936,941]`，`target="the entire text content of the document"` | 返回约 1,709 字符，覆盖标题、作者介绍和正文主体，接近完整逐字转录 |

该案例把总体 OCR 分项与工具行为直接连接起来。Crop 和 Atomic 都通过大范围空间约束保留了
页面结构；TGVF 虽请求“全部文本”，返回内容却偏向语义摘要。它与 latent 精确信息损失的
解释一致，但仍不能排除 policy 输出风格和停止行为的共同影响。

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
| TGVF 更适合语义定位与部分全局关系 | 地图、姿态、相对深度与 chair-horse 关系案例 | 语言 target 减少了显式 bbox 定位负担 | **支持行为结论；机制为推断** |
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

## 9. 论文主图与补充案例的组织建议

本页已经保留 `C1--C4`、`T1--T5`、`A1--A7` 和一个共享 OCR 案例，共 17 个带图片的
案例。论文正文不宜一次放入全部案例，建议正文定性图仍采用三列，每列一个主要成功案例和
一个边界案例：

| 列 | 成功案例 | 边界或失败案例 | 要表达的信息 |
|---|---|---|---|
| Crop | `HRBench4K/0159_0.jpg` 日晷年份 | `HRBench4K/0075_0.jpg` context limit | 像素保真强，但裁剪和序列开销可能失败 |
| TGVF | `HRBench4K/0178_0.jpg` 地图跨区域关系 | `VStarBench/0009_0.jpg` 颜色误判与重复 | 语义定位强，但精确读取和输出控制较弱 |
| Atomic | `MathVerse_MINI/0206_0.png` 方位角 | `BLINK/0144_0.jpg` 九宫格过度分解 | 定位后推理强，但局部调用可能破坏全局结构 |

共享 OCR 案例 `OCRBench_v2/0373_0.jpg` 可作为第四行，并列展示三者完整输出。其余
C2/C3、T1/T3/T4、A2--A6 适合放入补充材料，提供跨数据集复现，而不是只依赖正文中的单个
成功案例。

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

图片资产位于：

```text
docs/assets/prl25_qualitative_cases/
```

该目录共 17 张图。图片均来自上述 CoreDev-2511 benchmark 原图；高分辨率局部预览使用
`jpegtran` 做无损裁剪和熵编码优化，其他图片为原文件副本或无损 JPEG 优化，没有进行
生成式增删、重绘或内容修复。展示 crop 只用于让人类读者看清证据，模型实际请求的 bbox
仍以第 6 节轨迹表为准。正式公开仓库或论文前仍需按各上游 benchmark 的许可核对图片再分发
条件与署名要求。
