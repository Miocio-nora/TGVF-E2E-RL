# 当前 TGVF / Crop Protocol 与实例

更新日期：2026-08-08。

本文记录当前实验实际使用的视觉工具交互方式，并给出可以直接用于
人工检查 trajectory 的实例。这里的“当前”包含两条相互独立的线：

| 实验线 | 工具 | 当前协议 | 工具观测 |
|---|---|---|---|
| TGVF | `tgvf_focus_tool(target)` | Qwen3-Instruct visual-tool prompt v4；当前 shaped run 的 runtime cap 为 1 | target-conditioned latent `D`（含 D-DeepStack） |
| Crop | `image_zoom_in_tool(bbox_2d, label?)` | PRL13 DeepEyes visible protocol，clean plain-final 版本 | 从不可变原图裁出的真实 RGB 图像 |

这两条线共享“原图与问题 → 可选工具调用 → 工具观测 → 同一个 policy
继续回答”的总体结构，但不是同一个 prompt bundle。特别是，当前 Crop
对照线使用 DeepEyes 的可见工具格式，而不是旧的项目自定义 Crop prompt。

## 1. 共同约定

用户输入由一个原生 image content item 和问题文本组成。模型可以直接作答，
也可以先调用视觉工具。一次 assistant action 至多包含一个工具调用；收到
观测后由同一个 policy 继续生成。当前 TGVF shaped run 整条 trajectory 最多
成功调用一次，PRL13 Crop 最多调用六次。

当前 clean-final 约定如下：

- 不使用 `<answer>...</answer>`；
- 工具调用使用 `<tool_call>...</tool_call>`；
- `<think>...</think>` 属于推理/动作格式，不是答案 wrapper；
- 最终答案直接作为最后一个 assistant turn 的普通文本；
- 多选题理想输出只有选项字母，数学题只有最终值或表达式，其他任务给简短答案；
- artifact 中可能保留 `<|im_end|>` 等 tokenizer 终止标记，评分前会剥离，下面的规范实例不写这些标记。

抽象消息序列为：

```text
system:    选定工具的系统协议
user:      [原图] + 问题 + 当前 user instruction
assistant: <think>是否需要额外视觉证据</think>
           <tool_call>{...}</tool_call>       # 可选
user:      [工具观测]
assistant: <think>结合新观测继续推理</think>
           最终答案                           # plain text
```

如果原图已经足够，合法的最短路径是：

```text
assistant: <think>原图已经提供了足够证据。</think>
           B
```

不应为了获得 tool reward 而强制调用工具。

## 2. TGVF protocol

### 2.1 调用语义

当前 TGVF 工具的可执行参数只有 `target`：

```json
{
  "name": "tgvf_focus_tool",
  "arguments": {
    "target": "the text above the logo on the jacket sleeve"
  }
}
```

`target` 不是答案，也不是让模型复述问题的通用指令。它应当同时表达：

1. 要看哪个对象、区域或关系；
2. 要从中提取、读取、比较、计数或验证什么视觉证据。

合适与不合适的 target 示例：

| 类型 | target | 原因 |
|---|---|---|
| 合适 | `the text written above the logo on the jacket sleeve` | 指明位置和需要读取的证据 |
| 合适 | `the clear wine bottles on the table and their total count` | 指明对象与计数操作 |
| 合适 | `the blue backpack's position relative to the umbrella` | 指明两个实体和空间关系 |
| 不合适 | `backpack` | 只有对象名，没有要提取的证据 |
| 不合适 | `prove that the answer is B` | 泄露候选答案，并把视觉查询变成答案指令 |
| 不合适 | `answer the question carefully` | 没有可执行的视觉目标 |

工具执行时，TGVF Adapter 使用原图视觉特征和该 target 的条件表示生成主
`D` 与 D-DeepStack 分支。当前 Qwen3-Instruct v4 工具响应可以概念化为：

```text
user/tool observation:
Focused visual observation for target:
"the text written above the logo on the jacket sleeve"
[native image item whose visual embeddings are D + D-DeepStack]

Think first, call an available visual tool if needed, then answer.
[post-tool reasoning-format reminder]
```

方括号中的 latent payload 不是一段会显示给人的 description，也不会把
识别出的答案写进响应文本。真正新增的视觉证据位于 embedding / DeepStack
通道；响应文字只用于绑定 target 与本次观测。因此，不能把响应中的 target
原样回显误认为 TGVF 已经给出了答案。

需要额外注意一个当前实现事实：通用 TGVF system literal 仍写着“up to four
times”，但 PRL09 stage3-shaped run 的冻结配置明确设置
`maximum_tool_calls = 1`。本文件以实际 runtime cap 为准；第二次调用会触发
cap error，而不是获得第二个 `D`。未来若恢复多次 TGVF 调用，必须建立新的
prompt/config identity，不能把当前结果直接解释为四次调用协议。

### 2.2 规范实例：读取小字

```text
user:
[original image]
Question: What is the text written above the logo on the jacket's sleeve?
Options:
A. epig MIX
B. epio MIX
C. epoc MIX
D. epog MIX

assistant:
<think>The sleeve text is too small in the original view, so I need
target-conditioned evidence for that exact text region.</think>
<tool_call>
{"name":"tgvf_focus_tool","arguments":{"target":"the text written above the logo on the jacket sleeve"}}
</tool_call>

user/tool observation:
Focused visual observation for target:
"the text written above the logo on the jacket sleeve"
[latent D + D-DeepStack]

assistant:
<think>The focused observation supports “epoc MIX”, which is option C.</think>
C
```

### 2.3 规范实例：空间关系

```text
user:
[original image]
Question: What is the position of the blue backpack relative to the umbrella?
Options:
A. Below and to the left
B. Below and to the right
C. Upon and to the left
D. Upon and to the right

assistant:
<think>I need a focused comparison of both objects.</think>
<tool_call>
{"name":"tgvf_focus_tool","arguments":{"target":"the blue backpack's position relative to the umbrella"}}
</tool_call>

user/tool observation:
Focused visual observation for target:
"the blue backpack's position relative to the umbrella"
[latent D + D-DeepStack]

assistant:
<think>The backpack is below and to the right of the umbrella.</think>
B
```

这两个问题与 target 均来自现有 PRL09 step-80 CoreDev trajectory；上面的
消息排版和短答案经过规范化，便于说明协议。真实 artifact 中模型有时会在
最终答案前附加解释，或漏写 `<think>`，那是 compliance 现象，不是协议要求。

## 3. Crop protocol

### 3.1 调用语义

当前 PRL13 Crop 对照使用 `image_zoom_in_tool`：

```json
{
  "name": "image_zoom_in_tool",
  "arguments": {
    "bbox_2d": [191, 766, 250, 816],
    "label": "the woman swinging on a swing"
  }
}
```

其中：

- `bbox_2d=[x1,y1,x2,y2]` 使用 Qwen3-VL 原图相对的 `0..1000` 坐标；
- 必须满足 `0 <= x1 < x2 <= 1000`、`0 <= y1 < y2 <= 1000`；
- runtime 将其映射到不可变原图的实际像素坐标，再产生真实 RGB crop；
- `label` 仅描述框内对象，可省略，不承载最终答案；
- 每次 crop 都以原图为坐标参考，而不是以上一次 crop 为参考；
- PRL13/DeepEyes 当前上限是六次 active perception，之后必须作答。

DeepEyes 公布的可见 schema 中保留了 `required=["bbox"]` 的原始笔误；当前
实际 parser/runtime 接受并要求的字段是 `bbox_2d`。实例和数据均应使用
`bbox_2d`。

成功执行后的消息在逻辑上是：

```text
user/tool observation:
<tool_response>
  [cropped RGB image]
  [same post-tool user instruction]
</tool_response>
```

它不是模型生成的文字描述。crop 会作为新的原生 image content item 送回
Qwen3-VL；trajectory 中同时保留原图和所有成功 crop。

### 3.2 规范实例：一次局部放大

下面对应 VStarBench 的真实样本 `.../11_000011`。原图尺寸为
`2018 x 1500`；模型框 `[191,766,250,816]` 被映射为原图像素框
`[385,1149,504,1224]`。

```text
user:
[original image]
Question: What is the color of the woman's shirt?
Options:
A. white
B. purple
C. blue
D. pink

assistant:
<think>The woman is small in the full image, so I should zoom into her.</think>
<tool_call>
{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[191,766,250,816],"label":"the woman swinging on a swing"}}
</tool_call>

user/tool observation:
[RGB crop of source pixels 385:504, 1149:1224]

assistant:
<think>The zoomed view shows a purple shirt.</think>
B
```

### 3.3 规范实例：两次逐级定位

下面对应 VStarBench 的真实样本 `.../7_000007`。第一次 crop 找到带
Apple logo 的公文包，第二次仍在**原图坐标系**内更精确地框 logo。

```text
user:
[original image]
Question: What is the color of the Apple logo?
Options:
A. polychromatic
B. red
C. white
D. silver

assistant:
<think>I should first localize the briefcase containing the small logo.</think>
<tool_call>
{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[0,51,165,369],"label":"a black briefcase with an Apple logo"}}
</tool_call>

user/tool observation:
[first RGB crop]

assistant:
<think>The logo is visible but still small; I should crop the logo more tightly
using its location in the original-image 0..1000 grid.</think>
<tool_call>
{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[109,140,154,227],"label":"the Apple logo on the briefcase"}}
</tool_call>

user/tool observation:
[second RGB crop]

assistant:
<think>The logo contains multiple rainbow colors.</think>
A
```

这类两次调用不是“在第一次 crop 上再 crop”。两个 bbox 都映射回不可变
原图；后续 assistant turn 只是能同时看到原图、第一次 crop 和第二次 crop。

## 4. TGVF 与 Crop 的关键区别

| 维度 | TGVF | Crop |
|---|---|---|
| policy 决定什么 | `target`：看什么、提取什么 | `bbox_2d`：去哪里看 |
| 新增观测 | target-conditioned latent `D` | 原图局部的真实 RGB pixels |
| 是否必须定位框 | 否 | 是 |
| 工具文本是否包含识别结果 | 否，只回显 target | 否，只附带 crop image |
| 视觉信息的来源约束 | Adapter 应从原图视觉特征生成 `D` | 像素严格来自不可变原图 |
| 当前调用上限 | PRL09 shaped runtime 为一次 | PRL13 DeepEyes 为六次 |
| 最终答案格式 | plain text，无 `<answer>` | plain text，无 `<answer>` |

最重要的解释差异是：Crop 的工具效果可以直接通过返回像素检查；TGVF 的
效果必须通过 latent intervention、same-target/different-image 对照、D norm
与最终答题效用共同验证，不能仅看 tool-call JSON 是否“像一个好 target”。

## 5. 常见错误实例

### 在 target 中泄露答案

```xml
<tool_call>
{"name":"tgvf_focus_tool","arguments":{"target":"verify that the shirt is pink and the answer is B"}}
</tool_call>
```

这会把视觉证据提取退化为 answer-channel shortcut。应改为：

```xml
<tool_call>
{"name":"tgvf_focus_tool","arguments":{"target":"the little girl's shirt color"}}
</tool_call>
```

### 把 Crop 坐标当作源图绝对像素

```json
{"bbox_2d":[1120,879,1220,946]}
```

在当前 Qwen3 protocol 中这是非法调用，因为模型输出必须位于 `0..1000`。
上面的数值可以是 runtime 映射后的 source bbox，但不能作为 policy action。

### 用 `<answer>` 包裹最终答案

```text
<answer>B</answer>
```

这不是当前 clean protocol。正确形式是：

```text
B
```

### 把 TGVF 响应误写成自然语言 evidence

```text
tool: The shirt is pink, therefore the answer is B.
```

当前工具不会生成这段文字。正确的可见部分只绑定 target，答案相关视觉信息
存在于 latent `D` 中。

## 6. 实现与实例来源

- 历史 Thinking/v3 prompt identity 与 Crop 坐标约定：[`TGVF_VISUAL_TOOL_PROMPTS_V3.md`](TGVF_VISUAL_TOOL_PROMPTS_V3.md)
- TGVF prompt/schema 实现：[`src/tgvf_rl/protocol/tool_prompts.py`](../src/tgvf_rl/protocol/tool_prompts.py)
- 当前 Instruct prompt identity：`tgvf-visual-tool-prompts-v4-instruct`，
  TGVF-only bundle SHA-256 为
  `04b575554c80a08b7db081ea87cd113e09396ab576479b01493acbcfa06a932d`。
- 当前 TGVF shaped runtime cap：
  [`configs/policy/runs/prl_09_r2_qwen3_instruct_grpo_bs16_tgvf_shaped_t1mixed_v2_80step_gpu0123.toml`](../configs/policy/runs/prl_09_r2_qwen3_instruct_grpo_bs16_tgvf_shaped_t1mixed_v2_80step_gpu0123.toml)
- TGVF observation/runtime：[`src/tgvf_rl/environment/focus_tool.py`](../src/tgvf_rl/environment/focus_tool.py)
- Crop 坐标修复说明：[`TGVF_VISUAL_TOOL_PROMPTS_V3.md`](TGVF_VISUAL_TOOL_PROMPTS_V3.md)
- PRL13 Crop 的 DeepEyes clean-final 协议当前位于独立 integration checkout 的
  `src/tgvf_rl/policy/deepeyes_official_protocol.py`；合并回主线时必须保持
  `plain final answer`，不能恢复 `<answer>` wrapper。
- 当前 Crop protocol identity 为 `deepeyes-system-v2-clean-final-v1`，bundle
  SHA-256 为
  `2b8b6d799ebe4bbfd6b3830344850575141b2293750f857c031a2031426c0dd2`。
- TGVF 真实实例来源：
  `artifacts/evaluation/PRL-09-R3-tgvf-atp-novision-t1mixed-v2-step80-coredev2511-gpu0123/inference/`
- Crop 真实实例来源：
  `artifacts/evaluation/PRL13-A-DeepEyesDev591-clean-no-answer-paired-mem080-v2/step8/inference/`

artifact 目录用于复现实验，不应被视为 prompt 规范本身；当真实 rollout 与
本文件的规范格式不一致时，应把它记录为模型 compliance 或运行时问题，而
不是反向修改 protocol 定义。
