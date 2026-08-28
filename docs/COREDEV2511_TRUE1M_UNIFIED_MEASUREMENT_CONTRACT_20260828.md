# CoreDev-2511 true-1M 统一测量合同与重测台账

更新时间：2026-08-28 16:39 JST

状态：**合同已冻结；Crop S32/S80、TGVF S64 与 Atomic S16 的 true-1M 结果已
闭合，Original 与 No-Tool 仍在补测。本文是当前项目级唯一 true-1M 口径。** 在本文标为
`accepted` 之前，任何仅在配置中写有 `1,003,520`、但没有真实 processor/grid 证据的结果，均不得
进入统一主表。

## 1. 结论先行

PRL25 的 Crop、TGVF、Atomic 和 No-Tool RL 训练均使用
`image_max_pixels = 1,003,520`。`16,777,216 = 4096²` 是 Qwen3 fast image processor
保存的默认 `size.longest_edge`，不是这批 RL 训练允许的最大面积。

这里的“RL 训练”专指 PRL25 policy RL。TGVF/Atomic 所加载的 RP67 representation adapter
预训练使用 `image_max_pixels = 262,144 = 512²`；因此不能把 adapter 预训练与后续 policy RL
混称为同一个分辨率合同。两阶段都不是 `16,777,216`。

评测侧曾有两条独立缺陷：旧 Crop official-visible evaluator 与旧 No-Tool matched evaluator
均把像素覆盖值放在 processor 不读取的 flat kwarg 中，因而实际回退到 `16,777,216`。所以旧
Crop fixed-boundary S32/S80 与旧 No-Tool S0/S8/S16/S32 都不是 true-1M 结果。TGVF 与 Atomic
使用另一条已验证的 nested preprocessing 路径，不受该缺陷影响。Crop 已在顶层
`mm_processor_kwargs.size` 和 fixed action boundary 下重跑闭合。

### Stage golden result under the unified true-1M contract

下表是当前最对齐的三方结果：三条 policy RL 训练与评测都使用 `1,003,520`，评测均为
同一 `2,240` 条支持集、七项统计与 scorer。Crop 以 S32 作为当前性能代表，取代 S80 进入
golden 主表。单位为 `%`。

| Method | Macro* | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Crop S32 | 61.0706 | 73.2984 | 68.0000 | 63.3333 | 53.8604 | 46.4684 | 68.3333 | 54.2000 |
| TGVF S64 | 59.8086 | **74.3455** | 66.5000 | 65.5556 | 44.5446 | 44.9814 | **72.3333** | 50.4000 |
| Atomic S16 | **63.0827** | 71.7277 | **73.5000** | **66.1111** | **54.2720** | **51.3011** | 69.6667 | **55.0000** |

Atomic 在 Macro* 与七个组件中的五项最高；TGVF 在 VStar 与 MathVista 最高；Crop S32 的整体
表现更均衡，且高于同合同 Crop S80。Macro* 上，Atomic 比 Crop 高 `2.0121 pp`、比 TGVF
高 `3.2740 pp`，Crop 比 TGVF 高 `1.2619 pp`。这是统一输入预算下的端到端比较，不是严格单变量因果消融：
三者的工具协议和 checkpoint step 不同。RP67 adapter 预训练仍为 `512²`，但加载它的 TGVF/Atomic
policy RL 及本表评测均为 true-1M。

## 2. 固定术语

| 术语 | 唯一定义 |
|---|---|
| **true-1M** | `max_pixels = 1,003,520`；保持长宽比的最大表示面积，不是固定正方形 resize |
| **true-512** | `max_pixels = 262,144 = 512²` |
| **processor-default cap (≤16.7M)** | `size.longest_edge = 16,777,216 = 4096²`；它是失效历史运行的上限，不表示每张图都被放大到 16.7M |
| **Original raw-direct true-1M** | 原始 Qwen3-VL-8B-Instruct，官方 raw prompt，无 system prompt、无工具，seed42，仅把历史 direct 的像素上限改为 `1,003,520` |
| **No-Tool RL matched true-1M** | PRL25-F 的训练匹配 user-only no-tool prompt 与 direct-only loop，S0/S8/S16/S32 |
| **Crop fixed-boundary true-1M** | PRL25-B native RGB Crop，严格 `</tool_call>` action boundary，S32/S80 |
| **TGVF matched true-1M** | PRL25-C Pure TGVF、Frozen RP67，S64 |
| **Atomic (TGVF+Crop) matched true-1M** | PRL25-D Atomic Crop+TGVF、Frozen RP67，S16 |
| **Macro\*** | VStar、HRBench、BLINK-180、OCR EN/CN mean、MMMU-269、MathVista、MathVerse 五版本宏平均的七项无权平均 |

## 3. 统一主表合同

统一主表固定使用 CoreDev-2511、seed42、temperature 1、相同七项官方 scorer、BLINK 180
单图 coverage、MMMU 269 单图 coverage 和相同 Macro* 聚合。每行必须同时满足：

1. 配置声明 `max_pixels = 1,003,520`；
2. prompt expansion 与 vLLM decode 都必须使 Qwen3 的有效
   `size.longest_edge = 1,003,520`：本地 HF/训练 preprocessing 可使用
   `images_kwargs.size.longest_edge`；vLLM 0.12 必须使用可哈希的
   `mm_processor_kwargs.size.longest_edge`；direct VLMEvalKit 使用等价的逐图 `max_pixels` 绑定；
3. 对至少一张大于上限的真实 processor probe，保存 source area、`image_grid_thw`、visual
   token count 与 represented area，并验证 represented area 不超过 `1,003,520`；
4. `2,240/2,240` 个支持的单图任务完成推理，七个官方 slice 全部评分闭合；
5. accepted summary、coverage view、evaluation identity 与 completion receipt 均可追溯。

像素预算统一只消除输入面积这一项混杂。Original、No-Tool、Crop、TGVF 与 Atomic 的 prompt、
agent loop、工具 schema 和可调用工具仍不同，因此该表是**统一输入预算下的端到端比较**，不是
严格单变量方法消融。

## 4. 当前主表状态

| 主表行 | checkpoint | true-1M 状态 | 当前 Macro* |
|---|---|---|---:|
| Original raw-direct | base Qwen3-VL-8B-Instruct | `processor proof accepted; queued` | — |
| No-Tool RL | S32 | `pending rerun` | — |
| Crop | S32 | `accepted; stage golden` | 61.0706 |
| Crop | S80 | `accepted; diagnostic endpoint` | 59.6463 |
| TGVF | S64 | `accepted` | 59.8086 |
| Atomic (TGVF+Crop) | S16 | `accepted` | 63.0827 |

No-Tool S0/S8/S16/S32 全部补测，以保留训练动态；S32 是统一主表行，S0 是训练前控制行。

当前 accepted true-1M 结果如下，单位为 `%`：

| Method | Macro* | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Crop S32 | 61.0706 | 73.2984 | 68.0000 | 63.3333 | 53.8604 | 46.4684 | 68.3333 | 54.2000 |
| Crop S80 | 59.6463 | 73.8220 | 69.5000 | 59.4444 | 54.0426 | 44.9814 | 63.3333 | 52.4000 |
| TGVF S64 | 59.8086 | 74.3455 | 66.5000 | 65.5556 | 44.5446 | 44.9814 | 72.3333 | 50.4000 |
| Atomic S16 | 63.0827 | 71.7277 | 73.5000 | 66.1111 | 54.2720 | 51.3011 | 69.6667 | 55.0000 |

### 4.1 Crop true-1M 工具行为与边界审计

| Crop checkpoint | 至少一次成功工具调用的题数 | 成功调用率 | 实际调用次数 | `invalid_crop` | `tool_call_cap` |
|---|---:|---:|---:|---:|---:|
| S32 | 1,423 / 2,240 | 63.53% | 1,769 | 92 | 11 |
| S80 | 2,006 / 2,240 | 89.55% | 2,043 | 102 | 0 |

fixed action boundary 使合法 `</tool_call>` 在当轮终止，评分不再接受同轮工具请求后的 plain
final。结构审计另外发现 S80 中 **1 个**样例在一个已闭合调用后又生成了未闭合的第二个
`<tool_call>` opener。该次未被解析或执行为合法调用，也未重现旧的 answer-over-action 接受路径；
它仍是一个需要保留的格式边界，因此不将 S80 描述为“所有工具语法都完全闭合”。

可追溯审计键如下：

| Crop checkpoint | true-1M audit receipt identity SHA256 | `paired-summary.json` SHA256 |
|---|---|---|
| S32 | `0109f7f4f602106bf71bca50309019ae248962387dd7aa33f2b8e12c65042581` | `a3aa1befedb509e48b69edf4822d7e0467bca346b866da6fd78378d3265ea87d` |
| S80 | `fac02bd313a2c9786b93f7927dccb25403f7b6421aba3256e21d40a97757bbe9` | `715ae0aca71532804942ef5f301afb31c99080d8ec78768b9a10d2be8f448ae5` |

## 5. 有效 true-512 四方对照

下表中四行都已确认实际像素上限为 `262,144 = 512²`。Crop 使用 fixed-boundary V2 S80；
旧 boundary 的 `61.5591` 已被它取代。Atomic S16 评分已完成，不是 pending。单位为 `%`。

| Method / effective contract | Macro* | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original raw-direct@512 | 55.3556 | 50.7853 | 59.0000 | 65.5556 | 48.1848 | 39.0300 | 74.3333 | 50.6000 |
| Crop S80@512, boundary-fix V2 | **62.0967** | **86.9110** | **65.5000** | 64.4444 | **54.4299** | 45.7249 | 65.6667 | 52.0000 |
| TGVF S64@512 | 55.4067 | 53.4031 | 60.5000 | 62.7778 | 40.7642 | 46.4684 | 72.3333 | 51.6000 |
| Atomic S16@512 | 57.2762 | 57.0681 | 59.5000 | 61.6667 | 47.7713 | **48.3271** | 71.0000 | **55.6000** |

这是同像素上限下的端到端表，不是同 prompt 的严格方法消融。Original 使用 raw-direct
prompt，无 system prompt 与工具；Crop、TGVF 和 Atomic 分别使用各自训练匹配的 prompt、工具
schema 和 agent loop。因此跨行差值不得直接归因给某一工具或 RL。

## 6. 必须降级的历史结果

下列数值只保留为 `processor-default-cap historical`，不得再放入 true-1M 主表，也不得与 true-512
构成像素消融：

| 历史运行 | Macro* | 失效原因 |
|---|---:|---|
| Crop fixed-boundary S32 | 61.6699 | flat processor override 被忽略 |
| Crop fixed-boundary S80 | 59.1785 | flat processor override 被忽略 |
| No-Tool S0 | 64.4712 | flat processor override 被忽略 |
| No-Tool S8 | 66.1132 | flat processor override 被忽略 |
| No-Tool S16 | 66.9028 | flat processor override 被忽略 |
| No-Tool S32 | 66.6853 | flat processor override 被忽略 |

真实 Qwen3 processor 对 `2048×1536` 图像的审计结果为：

| 传参方式 | represented pixels | 结论 |
|---|---:|---|
| processor default | 3,145,728 | 未受 1M 限制 |
| flat `max_pixels=1,003,520` | 3,145,728 | 覆盖值被忽略 |
| direct HF：`images_kwargs.size.longest_edge=1,003,520` | 995,328 | 有效 true-1M；训练 preprocessing 使用 |
| vLLM 0.12：顶层 `mm_processor_kwargs.size.longest_edge=1,003,520` | 995,328 | 有效且 cache 可哈希；评测 runtime 使用 |

vLLM 0.12 不能直接使用 nested `mm_processor_kwargs.images_kwargs.size`：它虽然对裸 HF
processor 有效，但会在 vLLM processor cache 中触发 `unhashable dict`。本轮 Crop 首次启动在
第一条请求前即 fail-closed，未产生预测；现已改为顶层 `size`，须以重新启动后的 completion
receipt 为准。

历史 Original raw-direct 的 `55.3556` 使用有效 `262,144` 上限，是第 5 节 true-512 四方表的
Original 行，但不是统一 true-1M 表中的 Original 行。No-Tool S32 raw-direct@512 的 `54.3543`
同理只属于 512 合同。

## 7. 训练像素审计

训练链分为两层：

| 训练层 | 适用对象 | 生效像素上限 |
|---|---|---:|
| representation adapter 预训练 | RP67（随后供 TGVF/Atomic 加载） | `262,144 = 512²` |
| policy RL | PRL25-B/C/D/F | `1,003,520` |

RP67 的最终配置
`configs/representation/experiments/image_axis_grounding/rp67_qwen3_instruct_image_axis_grounded_2000_gpu01_finalize_step2000.toml`
（SHA256 `cbbfc146…`）绑定 `image_max_pixels = 262,144`；其 `metrics.jsonl` durable
completion 记录也保存同一值（run identity `0b53d04c…`，global step `2000`，adapter
SHA256 `13332865…`）。用历史 representation preprocessing 对 `2048×1536` RGB 复跑，得到
`image_grid_thw=[1,26,36]`、represented area `239,616`、merged visual tokens `234`，确认
RP67 adapter 预训练为 true-512。

PRL25-B/C/D/F 的冻结 run config 均绑定 `image_max_pixels = 1,003,520`。进一步按各自最终
launch provenance commit（B `08a9d8b4`、C `b87126ae`、D `017b5077`、F `7645fe4a`）回查
历史源码：在线 rollout 的 source image，以及 Crop/Atomic 的 crop observation，均调用
`preprocess_qwen3_rgb()`；该函数通过 Hugging Face 实际识别的
`images_kwargs.size.longest_edge=image_max_pixels` 先生成受限的
`pixel_values/image_grid_thw`，再把预展开视觉张量送入 vLLM。dataset prompt expansion 使用
同一有效绑定。

用训练模型的真实 Qwen3 processor 对 `2048×1536` RGB 复跑上述历史函数：processor 保存的
默认上限虽为 `16,777,216`，实际得到 `image_grid_thw=[1,54,72]`、represented area
`995,328`、merged visual tokens `972`，均满足 `1,003,520` 上限。因此这些 checkpoint 的
训练输入是 true-1M；`16,777,216` 仅是 override 缺失或失效时的 processor 默认值。

像素问题与 action-boundary 问题分开处理：Crop true-1M 重测同时要求 fixed
`</tool_call>` boundary；像素修复本身不改变 TGVF/Atomic 的 action boundary 结论。

## 8. 执行队列

- [x] 审计训练侧真实像素路径；确认 PRL25 RL 为 true-1M。
- [x] 撤销 Crop 与 No-Tool 旧 nominal-1M 身份。
- [x] 保留 TGVF S64 与 Atomic S16 accepted true-1M 结果。
- [x] Crop S32/S80：fixed boundary、有效顶层 `mm_processor_kwargs.size` cap、独立
  compile cache、`2,240/2,240` 推理、七项评分与 true-1M audit receipt 均已闭合。
- [ ] No-Tool S0/S8/S16/S32：顶层 `mm_processor_kwargs.size`、独立 compile cache，完成推理与评分。
- [ ] Original：raw-direct true-1M 的真实 processor probe 已通过（`2048×1536 → 1152×864`，
  represented area `995,328`）；七卡推理排在 Crop 后，相同官方 scorer 与 Macro* 聚合。
- [ ] 回填 Original 与 No-Tool 后，生成最终完整主表与 sub-benchmark；Crop 工具行为已回填。
- [x] 把 Crop accepted true-1M 结果同步到 NeurIPS workshop 报告；旧值仅保留在历史勘误区。

## 9. 文章 claim 边界

当前可报告的 stage-golden 观测是 Atomic S16 `63.0827` > Crop S32 `61.0706` > TGVF S64
`59.8086`。这一排序只适用于已闭合的三方 true-1M 端到端比较；Original 与 No-Tool
true-1M 仍未闭合，不得把它改写为完整方法榜单。同时，不能把跨 prompt、agent contract 与
checkpoint step 的差值直接归因给视觉工具。Atomic 是否进入正文主线，仍由 target-only 稳健性、
无偏 target 合格率与统一 true-1M 主表共同决定。
