# 从“可读 D”到“有用 D”

## TGVF Stage1 目标失配、因果约束与 RL 准入门槛

**副标题：** Qwen3-VL-8B-Instruct 迁移与 T1 数据筛选方案

**日期：** 2026-07-28（Asia/Tokyo）

**状态：** 设计决策与实施计划；尚不是 RP66/Instruct 的端到端 QA 实验结论

**项目：** TGVF End-to-End RL

---

## 执行摘要

### 核心判断

RP66 已经证明：在冻结 Qwen3-VL-8B-Instruct 的 teacher-forced
`evidence_description` readout 中，正确的 TGVF observation（下文记为
`D+`）能够被模型读取，并且通常优于 target-only、random D 和 matched wrong
D。它尚未证明：D 能够提高最终答案概率，或在自由生成中提高准确率。

原因不是 RP66 “训练得不够久”，而是现行 Stage1 的目标本身没有要求这一点。
当前 loss 只覆盖 evidence token；`short_answer`、答案终止 token 和自由生成均没有
梯度。因此，evidence NLL 和 Matrix CE 很好，并不能推出 QA 提升。历史
RP46/Thinking 的 18-row 直接替图 smoke 已经展示过这种断裂：correct-D 明显低于
image-only，也低于 matched-wrong-D。该结果不是 RP66/Instruct 的正式结论，但它足以
证明“现有代理目标可能成功而下游效用失败”。

### 固定决策

1. **把 Stage1 的主交付从 evidence readability 改成 answer utility。** 新增一个
   不含 gold evidence、与部署输入布局一致的 answer-level causal objective，要求
   correct D 在答案 NLL、答案相对 log-odds 和自由生成准确率上优于 zero/sham D 与
   matched-wrong D。
2. **第一轮冻结 Instruct reader，只训练 TGVF Adapter。** 这样测到的是 D 对原始
   Instruct 模型的 drop-in utility，避免语言模型与 D 共同形成无法解释的私有编码，
   也避免再次破坏原模型 reasoning 能力。
3. **T1 只表示“有难度与提升空间”，不表示“工具有用”。** 在 T1 后增加第二个
   utility gate；只有 `image + real D` 同时优于 image-only 和 sham/wrong D 的样本，
   才能成为 TGVF-RL 的 tool-positive 数据。
4. **Crop 复现与 TGVF 改进保持两条线。** Crop 是第一条可独立推进的复现线，也应
   作为 TGVF 的 oracle/positive control；TGVF reward 改进是第二条线，不能在 D 的
   utility 尚未过门槛时启动正式 RL。
5. **训练 loss 不能逻辑上保证未知数据精度。** 我们能建立的可靠保证是：切断信息
   旁路、采用 matched counterfactual loss，并把 held-out paired generation 结果设为
   不可绕过的 promotion gate。

### 建议的最小推进

- 先把已有 oracle-D evaluator 从 Thinking hard-pin 扩展到 RP66 Instruct，在
  256–500 个 held-out rows 上补齐 image-only、D-only、image+D、zero/sham-D 和
  matched-wrong-D 配对评估。
- 新增独立 answer-only supervision branch，不能把答案 label 直接接在现有 gold
  evidence transcript 后面。
- 做 E0/E1/E2 的 500-step 同预算对照；E2 为
  `answer NLL + correct-vs-zero/wrong margin + auxiliary losses`。只有 E2 通过本文门槛，
  才跑 2,000-step 正式训练并进入 TGVF-RL。

---

## 1. 背景、研究问题与边界

当前项目的大方向已经从 Qwen3-VL Thinking policy 转向
**Qwen3-VL-8B-Instruct**，并把数据筛选视为 RL 方法的一部分。实验工作分成两条线，
顺序如下：

1. **复现 Crop RL。** 在 coordinate/crop bug 已修复的基础上重跑有效的 Crop
   baseline。此前失败的 Crop run 不能用于判断 RL 方法优劣。
2. **改进 TGVF reward。** 只有在 Instruct、数据筛选、D utility 和优化门槛明确后，
   才比较新的 TGVF reward。

本报告回答的是两条线之前的共同基础问题：

> Stage1 产生的 D 是否真的给最终答题带来了可测的因果增益？如果当前约束不足，
> 应怎样改目标、数据筛选、架构约束和 RL 准入条件？

我们需要分别证明两个量：

- **替代效用（replacement utility）**

  `Delta_replace = Acc(Q, target, D+) - Acc(Q, target, D_control)`

  答案阶段完全移除原图以及任何已读取原图的缓存；视觉信息只能沿
  `image -> TGVF -> D -> answer` 传播。

- **增量效用（additive utility）**

  `Delta_add = Acc(image, Q, target, D+) - Acc(image, Q, target, control)`

  原图、问题、target、prompt 和 decoding 均固定，只替换工具 observation。若正式
  RL 使用 image+D 协议，`Delta_add` 是主指标；`Delta_replace` 仍用于证明 D 本身
  携带答案相关视觉信息。

控制项至少包括 zero/sham D、同图不同 target 的 real D、同 target 不同图的 real D
和 source-image shuffled D。zero D 主要是 OOD health control；真实 matched negative
才是主要因果对照。

---

## 2. 当前 Instruct Stage1 的审计结论

### 2.1 RP66 身份与训练配置

当前最终 Instruct representation artifact 是：

`artifacts/representation/RP-66-qwen3-instruct-balanced-t1-contextual-2000-gpu01`

| 字段 | RP66 设置 |
|---|---|
| Base reader | Qwen3-VL-8B-Instruct |
| 训练范围 | TGVF Adapter only；Qwen/vision/merger/decoder 冻结 |
| Target conditioning | contextual hidden state，last layer |
| Observation | main D + DeepStack layers 8/16/24 |
| 同图 group | K=4 |
| World size | 2 GPU |
| Global rows/update | 32 |
| Optimizer | AdamW，初始 LR `1e-4`，cosine，100-step warmup |
| 最大图像像素 | 262,144（约 512 x 512） |
| Steps | 2,000，状态 `complete` |
| 现行 objective | `1.0 L_matrix + 1.0 L_evidence + 0.1 L_norm` |
| Manifold loss | disabled，weight 0 |
| 数据 leakage warning | `warn_on_target_leakage=false` |

### 2.2 RP66 已经证明与尚未证明的内容

以下指标来自同一份 200-row、46-group 的 Instruct internal evaluation：

| 证据 | 数值 | 能说明什么 | 不能说明什么 |
|---|---:|---|---|
| Correct-D evidence mean NLL | 1.235 | Gold evidence 在 correct D 下可读 | 不是 answer NLL/accuracy |
| Target-only / random-D NLL | 4.114 / 4.033 | D 不只是空条件 | 仍可能是 teacher-evidence code |
| Correct 胜 target-only / random-D | 100% / 100% | Evidence readout 明显依赖 D | 不证明自由答题采用 D |
| Correct 胜 wrong-same / wrong-different | 95.5% / 88.6% | Target-specific readout 较强 | 不证明正确语义方向 |
| Correct 胜所有可用 controls | 85.5% | 代理任务区分明显 | 代理任务不等于 QA utility |
| Query retrieval Top-1 / Top-2 | 87.0% / 97.0% | 同图 target 区分较强 | 可能利用 identity code |
| MRR / mean diagonal gap | 0.929 / 2.434 | Matrix objective 已学习 | 不约束 answer 或 EOS |
| Native target-presence direction | 0.0%，n=36 | 暴露语义方向断裂风险 | 不是最终 QA accuracy |
| Target-presence continuation | 13.9% | 自由 continuation 很弱 | 小审计集不能作总 benchmark |
| Target-presence mean separation | -1.009 | 当前方向与期望相反 | 仍需更大、专门数据复核 |

表示张量均 finite，未发现 collapse；但 D/source mean token norm ratio 为 main 1.284、
layer-8 2.858、layer-16 2.303、layer-24 2.223。Norm 项保证了某种尺度约束，却没有
保证视觉语义或答题效用；部分 DeepStack branch 达到 source 的 2–3 倍，也说明只用
norm 不能排除强 soft-prompt 通道。

### 2.3 历史 RP46/Thinking 的下游反例性 smoke

此前 direct-D replacement smoke 使用的是最终 Thinking Stage1/RP46，不是 RP66。
18 个样本得到：

| Arm | Correct / Incorrect / Unclear | Accuracy 区间 |
|---|---:|---:|
| Image-only | 14 / 3 / 1 | 77.8%–83.3% |
| Zero-D replacement | 0 / 18 / 0 | 0% |
| Correct-D replacement | 6 / 11 / 1 | 33.3%–38.9% |
| Matched-wrong-D replacement | 12 / 5 / 1 | 66.7%–72.2% |

配对上，correct-D 对 image-only 为 1 win / 9 losses / 7 ties / 1 unclear；对 zero-D 为
6 / 0 / 11 / 1；对 matched-wrong-D 为 0 / 6 / 11 / 1。这说明该 D 不是纯零信号，
但“正确 D 比错误 D 更有助于答案”的关键方向没有成立。

这不能外推为 RP66/Instruct 的正式结果；它的作用是提供一个严格反例：
**evidence readability 与 target retrieval 成功，并不足以保证 QA utility。** 因此第一项
新实验必须是 RP66/Instruct 的完全配对 utility evaluation，而不是直接开始 RL。

---

## 3. 为什么当前 Stage1 无法保证 D 有用

### 3.1 Objective mismatch

现行目标为：

`L_old = 1.0 L_matrix + 1.0 L_evidence + 0.1 L_norm`

| 项 | 当前监督 | 实际约束 | 缺口 |
|---|---|---|---|
| `L_evidence` / `L_gen` | Teacher-forced `evidence_description` CE | D 帮助复述 teacher evidence | 不监督 short answer、EOS、自由生成 |
| `L_matrix` | 同图 KxK 对角 D 优于其它 target D | Target 区分与检索 | 可通过身份码或 teacher 文本码取巧 |
| `L_norm` | D 与 source visual 的历史 norm 关系 | 数值尺度 | 不约束语义或答案效用 |
| Answer loss | 无 | 无 | `short_answer` 没有梯度 |
| Causal contrast | 无 | 无 | 不要求 correct D 优于 zero/wrong D |
| Generation constraint | 无 | 无 | Teacher forcing 好不代表生成采用 D |

虽然 completed transcript 中存在 `evidence_description + short_answer`，label mapping 只把
evidence token 设为监督位置；答案、separator、EOS、prompt 和 tool wrapper 均为
ignore index。冻结 Qwen 不是错误；真正的问题是经过冻结 Qwen 反传给 Adapter 的梯度
只来自 evidence，不来自答案。

### 3.2 不能直接打开现有 transcript 的 answer labels

直接把现有 `short_answer` 从 `-100` 改为可训练，会产生两个严重 shortcut：

1. 答案位置已经看到了 gold `evidence_description`，而训练集中大量 evidence 直接含有
   答案文本；
2. 当前 original-image key block 只针对 evidence query 位置，不能证明答案阶段没有
   从原图或 image-conditioned KV 读取信息。

因此必须建立一个**独立的 answer-only transcript**：

`question/image -> oracle tool target -> D observation -> answer + EOS`

该分支不能包含 gold evidence。D-only 模式还必须创建 fresh context，移除原图 token、
原图 DeepStack 和所有已经读过原图的 pre-D KV。

### 3.3 Teacher-evidence answer overlap 与捷径风险

对当前 39,998-row train split 做大小写不敏感、答案 strip 后的精确子串审计：

| 检查 | 结果 | 解释 |
|---|---:|---|
| `short_answer` 出现在 `evidence_description` | 23,694 / 39,998 = 59.24% | 大量 evidence 直接含答案词；这是 supervision shortcut risk |
| `short_answer` 出现在 `target` | 195 / 39,998 = 0.49% | 简单 target 直接泄漏较少，但仍需 hard-fail |

这不等于所有样本都是“非法数据污染”；evidence 合理地包含事实答案是常见现象。但它
意味着 evidence reconstruction 很容易与“编码 teacher 文本/答案”混合，且新 answer
branch 绝不能继续看到 gold evidence。

新的数据检查还应覆盖数字、答案 alias、选择题选项、同义表达、value span 和 target
性质描述，不能只检查简单字符串。train/validation/test 应按 stable image UID、编辑
family 和模板切分；最终 test split 不参与数据筛选。

### 3.4 Target 可以向 D 写入语言码

当前 Adapter 先执行 target-to-visual attention，再让 visual query attend 到 enriched
target，并使用 target-side values 生成 `delta`，最后形成：

`conditioned_visual = visual_raw + gate * delta`

这允许 target 文本不仅选择视觉区域，还向每个视觉 token 写入语言/身份编码。现行
Matrix CE 会奖励“哪个 target”的可分性，但不强制 D 的 value 内容来自图像。

因此，counterfactual 必须固定 question/target，只改变图像中的局部值；若 loss-only
仍然出现 shortcut，则 Adapter v2 应把 target 限制为 spatial salience/gating 信号，让
value content 主要来自 source visual token，并限制 residual ratio，或向有效 Crop
observation 蒸馏。

---

## 4. 新的 answer-utility Stage1 目标

### 4.1 冻结 reader 下的答案分数

令冻结的 Instruct reader 为 `phi`，TGVF Adapter 为 `A_theta`：

`D_i = A_theta(V(image_i), H(question_i, target_i))`

在不含 gold evidence 的精确部署 context 中，定义正确答案（包含真实终止 token）的
长度归一化 log-prob：

`s_i(D) = (1 / |answer_i + EOS|) * log p_phi(answer_i + EOS | C_i(D))`

对于 MCQ，应使用正确选项相对其它选项的 log-odds；开放答案使用规范化 alias 与配对
错误值。裸 NLL 可作为基础指标，但不应是唯一分数。

### 4.2 三个主损失

1. **答案损失**

   `L_answer = - mean_i s_i(D_i+)`

2. **相对 zero/sham 的增益约束**

   `L_gain = mean_i softplus(m - [s_i(D_i+) - s_i(D_i_zero)])`

3. **相对真实 matched-wrong D 的反事实约束**

   `L_cf = mean_i,c softplus(m - [s_i(D_i+) - s_i(D_i,c_wrong)])`

推荐的第一版总目标为：

`L_new = 1.0 L_answer + 1.0 L_gain + 1.0 L_cf`

`        + 0.25 L_evidence + 0.25 L_matrix + 0.1 L_norm`

上述权重是 smoke 起点，不是最终默认值。实现后必须比较各项 value、Adapter gradient
norm 和有效 row count，再调整权重与 margin `m`。现有 evidence 与 Matrix 目标降为
辅助项，用于保留 readout 可解释性和 target specificity，而不是作为 promotion 的
主依据。

### 4.3 两种必须同时存在的 context

| Context | 输入合同 | 证明的问题 |
|---|---|---|
| D-only fresh context | 无原图 token、原图 DeepStack、image-seen pre-D KV | D 自身是否足以承载答案相关信息 |
| Native image + D | 与部署完全相同；原图固定，只原子替换 D | D 在真实 pipeline 中是否有边际增益 |

如果 D-only 通过而 image+D 不通过，问题更可能是 reader/tool-context integration 或 D
被原图淹没；如果两者都失败，问题在 Adapter 表示、数据或目标，不应让 RL 来“救”。

### 4.4 反事实 controls 的设计合同

主要 negatives 应包含：

- 同图、不同 target 的 real D：测 target specificity；
- 同 target、不同图且答案不同的 real D：测视觉 grounding；
- 同 question/target、只改变局部视觉值的 value-flip pair：最强的 grounding 约束；
- shuffled source image：测视觉来源；
- target-only/no-image 产生的 D：测 target/Hq shortcut；
- zero D：只作为 OOD health control。

每次交换必须原子替换 main D 和 DeepStack layers 8/16/24；shape、grid、M-RoPE、mask、
dtype、token 数和 prompt 均保持一致。negative 在自己的配对样本中也应是 positive，使用
对称 2x2/矩阵目标，防止模型学习“坏 lane marker”。另应训练或评估一个只看 D 的
correct/wrong lane classifier；如果它不看 question/image 就能高于随机识别 lane，说明
存在 artifact shortcut，数据不能进入正式训练。

---

## 5. Crop、T1 与工具效用数据筛选

### 5.1 T1 不是 utility label

当前 Instruct T1 规则是对每个 candidate 做 8 次 full-image direct rollout：

- 0/8 correct：排除，通常过难或模型无能力；
- 8/8 correct：排除，通常过易、没有训练 headroom；
- 1–7/8 correct：保留为 T1。

T1 只测“当前 policy 的直接答题难度与不稳定性”，不测 D 或 Crop 是否真的有帮助。
因此不能把 T1 自动视为 `tool_needed`。

### 5.2 新增 T1U utility gate

对 T1 candidate 使用相同 prompt、相同 seed/decoding 的配对 arms：

| Arm | 条件 | 用途 |
|---|---|---|
| A | Image-only direct | 原始基线 |
| B | Image + real correct D | 实际 TGVF 增量效用 |
| C | Image + zero/sham D | 工具协议/额外 token 对照 |
| D | Image + matched-wrong real D | 语义因果对照 |
| E | Image + oracle crop（可用时） | 工具上界与 positive control |
| F | D-only correct/zero/wrong | D 自身充分性诊断 |

tool-positive 样本必须满足 `B > A` 且 `B > C/D`；E 提升而 B 不提升，说明 row 是可被
视觉工具帮助的，但当前 TGVF 不合格。B 无增益或有害的 rows 不应简单丢弃，可作为
no-tool negatives，帮助 RL 学会不调用工具。

低成本筛选可定义：

`u_i = min[s_i(D+) - s_i(D_zero), s_i(D+) - s_i(D_wrong)]`

先用 teacher-forced answer log-prob 筛选，再用 free generation 复核。不能在最终 test
split 上筛选或选择 checkpoint。

### 5.3 Crop 线如何帮助 TGVF

Crop 不只是另一种工具 baseline，还能成为 TGVF 的诊断与教师：

- **Positive control：** 若 oracle/修复后的 crop 都不能提高 Instruct，则该 row 不应
  作为工具正例。
- **Upper bound：** crop 有增益而 D 没有，定位为 Stage1/TGVF 表示问题。
- **Distillation teacher：** 在 `crop > direct` 的 rows 上，可蒸馏 crop-conditioned
  answer logits/hidden states 到 D-conditioned 行为。
- **Visual-manifold teacher：** 必要时让 D 对齐真实 crop observation 的视觉表示，而
  不是只对齐 teacher 文本。

这样能把“先复现 Crop”和“再改进 TGVF reward”两条线连接起来，但不混淆各自结论。

---

## 6. 训练范围与架构升级顺序

### 6.1 第一阶段：只证明 D 的 drop-in utility

- **训练：** TGVF main Adapter + 三个 D-DeepStack branches；
- **冻结：** Qwen3-VL-8B-Instruct vision tower、merger、decoder、embedding、LM head；
- **target Hq：** 使用 answer-free、sanitized 的 question+target；尽量使用 text-only
  fresh state，或在 counterfactual pair 中保持 Hq 完全一致并 detach；
- **评估：** answer log-prob、D-only generation、native image+D generation、no-harm。

冻结 reader 是识别 D 本体效用的实验控制，不是永久限制。如果一开始同时训练 Qwen，
LM 与 D 可能共同发明私有编码；随后无法判断是 D 对原始 reader 有用，还是共同训练
只适配了一个新接口。

### 6.2 Stage1.5：仅在 D-only 通过、image+D 失败时

此时可以：

- 冻结已经通过 D-only gate 的 TGVF；
- 只训练 very narrow decoder reader-LoRA；
- 使用 real/sham-D pair loss、direct replay 和 KL retention；
- 加一个同数据、同预算的 direct-only LoRA control。

如果 D-only 自身未通过，不开启 reader-LoRA，也不 full-tune vision tower。首轮 RL pilot
同样冻结 TGVF，只训练 decoder LoRA；只有 utility 已被证明后，才考虑 Adapter+LM 联合
RL，并采用交替更新和 observation versioning。

### 6.3 Adapter v2 的触发条件

如果加入 answer/counterfactual loss 后仍出现 target copy、lane artifact 或 DeepStack
delta 过强，再修改结构：

- target 只控制 spatial salience/gate，不提供 unrestricted value；
- D 的 value content 必须主要来自 source visual tokens；
- 限制或正则化 `||delta|| / ||visual_raw||`；
- 对有效 crop 的视觉 tokens、decoder logits 或 hidden state 做蒸馏。

这应是 loss 与干净 counterfactual 已验证仍失败后的结构消融，而不是第一步同时改动。

---

## 7. Stage1 Promotion Gate

| Gate | 必须满足 | 未通过时 |
|---|---|---|
| G0 数据/实现完整性 | Held-out；无 answer/evidence 泄漏；correct/zero/wrong 几何一致；DeepStack 原子交换；部署同构 | 修实现，不能解释模型效果 |
| G1 answer likelihood | Correct D 相对 zero 和 matched-wrong 的 paired delta 均为正；bootstrap 95% CI 下界 > 0 | 不进入长训练 |
| G2 D-only generation | `Acc(D+) > Acc(D_zero)` 且 `Acc(D+) > Acc(D_wrong)`；correct-only wins 多于 reverse wins；termination/parse 不恶化超过 2 pp | D 本体未证明有用 |
| G3 additive generation | `Acc(image+D+) > Acc(image+D_zero/wrong)` 且高于 image-only；paired CI 为正 | 不进入 TGVF-RL |
| G4 no-harm/稳健性 | 原本 direct-correct 子集退化不超过 1 pp；OCR/chart/counting/spatial 等切片方向一致；不同 seed/decoding 一致 | 继续筛数据或改接口 |
| G5 readout retention | 若保留可解释性目标，evidence readout 相对 RP66 下降不超过约 5% | 调整辅助项权重 |
| G6 RL 准入 | G1–G5 全部通过；随后做 real-tool vs sham-tool/no-tool matched RL | TGVF-RL 暂停，Crop 线可继续 |

建议正式 generation gate 使用至少 500 个 held-out rows；准确率使用 paired bootstrap CI，
同时报告 McNemar correct-only/reverse-only。64-row smoke 只判断方向，不能 promotion。
作为预注册的实用阈值，可要求：

- real D 对 zero、matched-wrong 的 paired likelihood win rate > 60%，且 95% CI 下界
  高于 50%；
- generated accuracy 相对 direct 和 sham/wrong 至少提升 2 pp，paired CI 不跨 0；
- real-vs-wrong 的 win/loss ratio 至少约 1.5；
- zero 或 wrong 与 real 持平/更好时，直接停止 TGVF-RL。

数值阈值应在看正式 test 结果前固定；如果样本量使 2 pp 无法可靠识别，应先扩展
held-out utility split，而不是放宽结论。

---

## 8. 最小实验矩阵、时间与停止条件

### 8.1 先评估，再训练

现有 `oracle_d_utility.py` 已包含 image-only、zero-D replacement、correct-D replacement、
matched-wrong-D replacement、D-only 和 image+D 等主要 arms，并已把模型可见字段与
ground truth 分开。第一项工程改动是解除其 Thinking-only hard-pin，补齐 Instruct
dialect/EOS，并对 RP66 运行 256–500-row paired evaluation。

在这一步结束前，不应把 RP66 evidence 指标解释为 TGVF 工具有效，也不启动正式
TGVF-RL。

### 8.2 Objective ablation

| Cell | 初始化与预算 | Objective | Qwen | 决策用途 |
|---|---|---|---|---|
| E0 | 统一 baseline | 现有 Matrix/evidence/norm | Frozen | 当前代理目标对照 |
| E1 | 同初始化、同数据、同 steps | E0 + answer NLL | Frozen | 判断直接答案梯度是否足够 |
| E2（推荐） | 同初始化、同数据、同 steps | E1 + correct-vs-zero/wrong margin + local-value CF | Frozen | 因果 utility 主方案 |
| E3（条件触发） | E2 通过 D-only 后 | E2 + narrow reader-LoRA | LoRA | 只诊断 additive interface |

可以先用 RP66 warm-start 做 64-row、100–300-step implementation smoke，每 50 step 看
paired direction；随后 E0/E1/E2 必须采用统一初始化、seed、数据和 500-step 预算作可比
消融，胜者再跑 2,000 steps。warm-start smoke 不能代替正式 from-common-init 对照。

按 RP66 约 4.4–4.8 s/step 的历史速度，若 E2 的 answer + zero + wrong forward 使训练
core 约为旧版 2.25 倍，则粗略估计：

- E2 500-step：约 1.2–1.5 小时；
- E2 2,000-step：约 5–6 小时；
- 另加 held-out free generation 和配对统计时间。

该估计需在新 streaming/VJP 实现完成后的前 20–50 step 重新校准，不能作为机器预约
保证。

### 8.3 诊断决策树

```text
image + oracle crop/textual evidence 是否提升？
  ├─ 否：target/data/model 本身不适合工具增强 -> 改筛选，不训练 TGVF 正例
  └─ 是：继续
       │
       ├─ correct D 是否优于 zero/wrong D？
       │    ├─ 否：Stage1 objective / Adapter grounding 失败 -> E2 或 Adapter v2
       │    └─ 是：继续
       │
       └─ image + D 是否优于 image-only？
            ├─ 否：reader/协议整合失败 -> 条件触发 Stage1.5
            └─ 是：D utility 过关 -> 才进入 TGVF-RL
```

---

## 9. RL 阶段应怎样使用这项结论

当前 pilot reward 的结构是 answer + format + conditional tool bonus；只要最终答对且得到
一个成功 observation，就可能获得工具奖励。这奖励的是“调用与答对同时发生”，不是
“D 导致了答对”。旧 Stage3 同样存在 tool-call 过度激励：旧实验的精度提升主要不能
归因于 TGVF observation 本身，因为当时 vision/TGVF 冻结，policy 只更新语言侧，并且
post-hoc tool subset 没有显示稳定增益。

建议移除无条件 tool-success bonus，引入小 call cost，并使用实测 utility：

`R = R_answer + beta * I(tool) * u_i - c_call`

其中 `u_i` 来自 held-out/离线 real-D 相对 sham/wrong-D 的实际增益。预算允许时，在部分
rollout 上做 paired replay：

`R_tool = beta * [r(real D) - r(matched sham D)] - c_call`

若 real 与 sham 都答对，不给“因果工具增益”奖励。

进入 RL 后至少做三个同数据、同 batch、同低 LR 的控制：

| RL Cell | Observation | 目的 |
|---|---|---|
| R0 | No-tool/direct | 普通 RL 与数据本身增益 |
| R1 | Real TGVF D | 完整工具增益 |
| R2 | Sham/zero/wrong D tool | 工具协议、额外 token 和调用奖励控制 |

`R1 - R2` 才是 tool-specific gain。首轮建议沿用已讨论的
`BS16 x n8 x 80 steps` 验证几何，并将 decoder-LoRA LR 从缺乏依据的 `1e-5` 降到有
历史参照的保守量级（优先 `1e-6`，再做小网格）；不要同时改变 full/LoRA、分辨率、
reward 和数据，以免失去归因。

---

## 10. 实施落点

### 10.1 可以复用的现有实现

- `src/tgvf_rl/representation/training/oracle_d_utility.py`
  - 已定义主要 paired arms；
  - prompt 不向模型暴露 ground truth；
  - 需要解除 Thinking-only model guard，支持 RP66 Instruct。
- `src/tgvf_rl/representation/training/streaming.py`
  - 已能让冻结 8B Qwen 通过 D boundary 做 VJP，再将梯度传回 Adapter；
  - 可扩展为 answer/counterfactual score，避免保留完整 reader graph。
- `src/tgvf_rl/representation/training/qwen3_counterfactual.py`
  - 可复用 main+DeepStack 原子替换和 matched counterfactual 合同。
- `src/tgvf_rl/representation/training/internal_evaluation.py`
  - 可复用 readout、health 和 paired metrics 基础设施。

### 10.2 必须新增或修改的合同

1. 新的 answer-only supervision schema，不污染现有
   `CanonicalEvidenceSupervision`；
2. native pipeline 的 answer-only row builder，明确 D-only fresh 与 image+D 两种路径；
3. streaming answer/counterfactual VJP 与全局有效 row reduction；
4. objective-v4 新权重、margin、EOS 和配置 identity；
5. Instruct oracle evaluator 与 periodic/final promotion report；
6. T1 后的 utility/T1U selector 与 resume-safe manifest；
7. 泄漏 hard-fail、matched geometry proof 和 lane-classifier audit；
8. RL reward 中 real-vs-sham differential 或离线 `u_i`。

---

## 11. 最终方向与执行顺序

下一阶段不再把 evidence reconstruction 当作 TGVF 成功标准。Stage1 的正式交付必须是：

> 在完全配对、无 gold teacher evidence、部署同构的条件下，correct D 对答案的增益
> 显著高于 zero/wrong D，并且在 image+D 条件下高于 image-only。

建议执行顺序固定为：

1. **RP66/Instruct utility baseline：** port evaluator，N=256–500；
2. **Answer-only objective implementation：** fresh D-only + native image+D；
3. **E0/E1/E2 500-step ablation：** 同初始化、数据、seed 与预算；
4. **Promotion：** 仅 E2 通过 G0–G5 才跑 2,000-step；
5. **T1U 数据筛选：** 将 tool-positive 与 no-tool/harmful rows 分开；
6. **Crop positive control：** 复现线独立继续，并为 D 提供 oracle/teacher；
7. **RL causal ablation：** R0 no-tool、R1 real-D、R2 sham-D，低 LR、小步数；
8. **条件升级：** 只有 D-only 通过、additive 失败时做 reader-LoRA；只有 loss/数据仍
   失败时做 Adapter v2。

如果这套门槛不成立，改进 TGVF reward 没有明确的有效工具通道可优化；RL 更可能放大
调用偏好、协议长度或幻觉，而不是学习真实的视觉增益。

---

## 附录 A：证据索引

### RP66 配置与结果

- `configs/representation/qwen3_instruct_balanced_t1_contextual_2000step_gpu01.toml`
- `artifacts/representation/RP-66-qwen3-instruct-balanced-t1-contextual-2000-gpu01/metrics.jsonl`
- `artifacts/representation/RP-66-qwen3-instruct-balanced-t1-contextual-2000-gpu01/internal_evaluation_v4_golden_audited_grounding_report.json`

### 当前训练与 Adapter 实现

- Evidence-only labels：
  `src/tgvf_rl/representation/training/transcript.py:420-470,543-587`
- Original-image evidence key block：
  `src/tgvf_rl/representation/training/streaming.py:1329-1345`
- Native Instruct transcript：
  `src/tgvf_rl/representation/training/native_pipeline.py:259-295`
- Current objective/loss：
  `src/tgvf_rl/representation/training/objective.py`，
  `src/tgvf_rl/representation/training/losses.py`
- Target-to-visual-to-target-valued delta：
  `src/tgvf_rl/representation/adapter.py:172-194`
- Oracle D utility evaluator：
  `src/tgvf_rl/representation/training/oracle_d_utility.py`

### RL 与数据筛选

- T1 规则：`docs/POLICY_RL_DATA_SELECTION.md`
- Selector：`src/tgvf_rl/data/policy_selection.py:341-420`
- Current reward：`src/tgvf_rl/rewards/pipeline.py:80-135`
- 旧/新 RL 审计：`docs/POLICY_RL_FUTURE_DIRECTION_20260728.md`

## 附录 B：结论边界

- RP66 readout 指标是 Instruct 结果；历史 18-row direct-D replacement 是 RP46/Thinking
  结果，两者不可混写。
- 36-pair target-presence 诊断是 failure detector，不是标准 QA benchmark。
- 本报告的 objective 权重、margin 和 promotion 数值是预注册起点，需要通过统一预算
  ablation 验证，但不能在看到最终 test 后反向选择。
- 尚未完成 RP66 的正式 image-only vs real-D vs sham/wrong-D free-generation 评估；因此
  当前最准确的表述是“D utility 未被证明”，而不是“RP66 D 已被证明无效”。
- 报告没有建议让 RL 修复未通过 gate 的 D；这是一项有意的实验归因约束。
