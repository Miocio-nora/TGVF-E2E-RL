# CoreDev-2511 true-1M 统一测量合同与重测台账

更新时间：2026-08-28 14:42 JST

状态：**合同已冻结；结果补测进行中。本文是当前项目级唯一 true-1M 口径。** 在本文标为
`accepted` 之前，任何仅在配置中写有 `1,003,520`、但没有真实 processor/grid 证据的结果，均不得
进入统一主表。

## 1. 结论先行

PRL25 的 Crop、TGVF、Atomic 和 No-Tool RL 训练均使用
`image_max_pixels = 1,003,520`。`16,777,216 = 4096²` 是 Qwen3 fast image processor
保存的默认 `size.longest_edge`，不是这批 RL 训练允许的最大面积。

评测侧曾有两条独立缺陷：旧 Crop official-visible evaluator 与旧 No-Tool matched evaluator
均把像素覆盖值放在 processor 不读取的 flat kwarg 中，因而实际回退到 `16,777,216`。所以旧
Crop fixed-boundary S32/S80 与旧 No-Tool S0/S8/S16/S32 都不是 true-1M 结果。TGVF 与 Atomic
使用另一条已验证的 nested preprocessing 路径，不受该缺陷影响。

## 2. 固定术语

| 术语 | 唯一定义 |
|---|---|
| **true-1M** | `max_pixels = 1,003,520`；保持长宽比的最大表示面积，不是固定正方形 resize |
| **true-512** | `max_pixels = 262,144 = 512²` |
| **processor default / actual 16.7M** | `size.longest_edge = 16,777,216 = 4096²`；只用于标注失效历史运行 |
| **Original raw-direct true-1M** | 原始 Qwen3-VL-8B-Instruct，官方 raw prompt，无 system prompt、无工具，seed42，仅把历史 direct 的像素上限改为 `1,003,520` |
| **No-Tool RL matched true-1M** | PRL25-F 的训练匹配 user-only no-tool prompt 与 direct-only loop，S0/S8/S16/S32 |
| **Crop fixed-boundary true-1M** | PRL25-B native RGB Crop，严格 `</tool_call>` action boundary，S32/S80 |
| **TGVF matched true-1M** | PRL25-C Pure TGVF、Frozen RP67，S64 |
| **Atomic matched true-1M** | PRL25-D Atomic Crop+TGVF、Frozen RP67，S16 |
| **Macro\*** | VStar、HRBench、BLINK-180、OCR EN/CN mean、MMMU-269、MathVista、MathVerse 五版本宏平均的七项无权平均 |

## 3. 统一主表合同

统一主表固定使用 CoreDev-2511、seed42、temperature 1、相同七项官方 scorer、BLINK 180
单图 coverage、MMMU 269 单图 coverage 和相同 Macro* 聚合。每行必须同时满足：

1. 配置声明 `max_pixels = 1,003,520`；
2. prompt expansion 与 vLLM decode 都使用 nested
   `images_kwargs.size.longest_edge = 1,003,520`，或 direct VLMEvalKit 中等价的逐图
   `max_pixels` 绑定；
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
| Crop | S32 | `inference running (GPU 4–7)` | — |
| Crop | S80 | `inference running (GPU 0–3)` | — |
| TGVF | S64 | `accepted` | 59.8086 |
| Atomic | S16 | `accepted` | 63.0827 |

No-Tool S0/S8/S16/S32 全部补测，以保留训练动态；S32 是统一主表行，S0 是训练前控制行。

当前两个 accepted true-1M 结果如下，单位为 `%`：

| Method | Macro* | VStar | HR | BLINK-180 | OCR mean | MMMU-269 | MathVista | MathVerse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TGVF S64 | 59.8086 | 74.3455 | 66.5000 | 65.5556 | 44.5446 | 44.9814 | 72.3333 | 50.4000 |
| Atomic S16 | 63.0827 | 71.7277 | 73.5000 | 66.1111 | 54.2720 | 51.3011 | 69.6667 | 55.0000 |

## 5. 必须降级的历史结果

下列数值只保留为 `actual 16.7M historical`，不得再放入 true-1M 主表，也不得与 true-512
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
| nested `images_kwargs.size.longest_edge=1,003,520` | 995,328 | 有效 true-1M |

历史 Original raw-direct 的 `55.3556` 使用有效 `262,144` 上限，仍可作为 raw-direct@512
历史参考，但不是统一 true-1M 表中的 Original 行。No-Tool S32 raw-direct@512 的 `54.3543`
同理只属于 512 合同。

## 6. 训练像素审计

PRL25-B/C/D/F 的 run config 均声明 `image_max_pixels = 1,003,520`。训练数据的 source-image
prompt expansion 使用 nested `images_kwargs.size.longest_edge`；在线 rollout 使用预展开的
image embeddings 与已绑定的 `image_grid_thw`。Crop observation 与 TGVF preprocessing 也在
同一上限下物化。因此 checkpoint 中保存的 processor 默认 `16,777,216` 不能被解读为训练
实际输入面积。

像素问题与 action-boundary 问题分开处理：Crop true-1M 重测同时要求 fixed
`</tool_call>` boundary；像素修复本身不改变 TGVF/Atomic 的 action boundary 结论。

## 7. 执行队列

- [x] 审计训练侧真实像素路径；确认 PRL25 RL 为 true-1M。
- [x] 撤销 Crop 与 No-Tool 旧 nominal-1M 身份。
- [x] 保留 TGVF S64 与 Atomic S16 accepted true-1M 结果。
- [ ] Crop S32/S80：fixed boundary、nested pixel cap、独立 compile cache；14:40 JST 已并行启动
  true-1M 推理，待推理与评分闭合。
- [ ] No-Tool S0/S8/S16/S32：nested prompt/decode、独立 compile cache，完成推理与评分。
- [ ] Original：raw-direct true-1M 的真实 processor probe 已通过（`2048×1536 → 1152×864`，
  represented area `995,328`）；七卡推理排在 Crop 后，相同官方 scorer 与 Macro* 聚合。
- [ ] 回填完整七项主表、四步 No-Tool 动态、sub-benchmark 与工具行为统计。
- [ ] 把 accepted 结果同步到 NeurIPS workshop 报告；旧值继续保留但只放历史勘误区。

## 8. 文章 claim 边界

在完整 true-1M 表闭合前，不报告哪一种方法总体最优。闭合后可以比较统一输入预算下的端到端
性能，并寻找 TGVF/Atomic/Crop 在 sub-benchmark 上的相对优势；不能把跨 prompt 与 agent
contract 的差值直接归因给视觉工具。Atomic 是否进入正文主线，仍由 target-only 稳健性、无偏
target 合格率与统一 true-1M 主表共同决定。
