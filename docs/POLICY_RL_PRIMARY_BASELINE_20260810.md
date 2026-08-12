# Policy RL 主对比基线：Original / Crop / TGVF（Step 0 / Step 8）

日期：2026-08-10

状态：`PRIMARY BASELINE / FROZEN SNAPSHOT`

> **2026-08-12 measurement update：** 本文的模型、训练配置、checkpoint 与
> artifact 身份仍有效；但第 1、8、10 节的 headline 聚合使用了 HRBench cycle 0
> 和 OCR Chinese-only，已由
> `docs/POLICY_RL_COREDEV2511_MEASUREMENT_CONTRACT_AND_BASELINES_20260812.md`
> 的 HR `Average/all`、OCR EN/CN mean 和 canonical Macro* 口径取代。后续不得再把
> 本文的 `53.83 / 56.57 / 58.77` 当作统一主均值。

Baseline ID：`POLICY-RL-PRIMARY-BASELINE-20260810-v1`

模型：`Qwen3-VL-8B-Instruct`

评测：`CoreDev-2511`，VLMEvalKit commit `7055d301`

用途：后续 Crop 与 TGVF policy-RL 实验的主要对比基线

## 1. 结论

在共同单图支持集上，当前五个 arm 的七项等权诊断均值为：

| arm | 诊断均值（%） |
|---|---:|
| Original，无工具 | 53.83 |
| Crop Step 0 | 56.57 |
| TGVF RP66 Step 0 | 53.85 |
| Crop Step 8，clean-final | **58.77** |
| TGVF RP66 Step 8 | 51.10 |

当前最可靠的判断是：

- clean-final Crop RL 是净正向的：`Step 0 -> Step 8 = +2.20 pp`，七项中四项提升；
- 当前 TGVF RL 是净负向的：`Step 0 -> Step 8 = -2.75 pp`，七项中五项下降；
- Crop Step 8 在七个共同支持 headline 指标上全部高于 TGVF Step 8；
- 这说明当前 **TGVF 的 policy-RL 更新过程不健康**，但不等价于证明 RP66/TGVF
  在 RL 前没有视觉效用；TGVF Step 0 已包含训练完成的 RP66，必须把 Stage 1
  初始化与后续 RL 更新分开解释。

本文冻结这组结果及其配置口径。后续实验不得用 PRL13 的历史 Step 8、full-set
zero-padding 分数或不同 prompt 的 raw baseline 替换本文主表中的 arm。

## 2. 比较问题与因果边界

本基线回答三个不同问题：

1. 原始 Instruct 模型在没有视觉工具时的端到端能力是多少；
2. 接入 Crop 或 RP66-TGVF 后，训练前的工具 pipeline 能力是多少；
3. 在各自 pipeline 内，8 个 optimizer update 带来了什么变化。

严格程度并不相同：

- `Original` 是 raw direct 参考，不使用工具 prompt，因此不是 Crop/TGVF 的严格
  paired control；
- `Crop Step 0` 是纯 base Qwen + clean Crop protocol；
- `TGVF Step 0` 是同一 base Qwen + 已训练的 RP66 Step 2000 + TGVF protocol，
  并非 raw base；
- 同一工具内部的 `Step 0 -> Step 8` 是解释 RL 效果的首要比较；
- `Crop vs TGVF` 是完整 treatment 比较。两者的 action、observation 与 prompt
  天然不同，不应解释成仅替换了一块像素/latent payload。

## 3. 五个 arm 的冻结身份

| arm | policy / 工具状态 | 冻结身份 |
|---|---|---|
| Original | 官方 Instruct；无工具、无自定义 system | `PRL-04-R2-raw-instruct-coredev2511-gpu4567-r4` |
| Crop Step 0 | base Qwen；clean-final Crop protocol | PRL13 clean Step 0，作为 PRL14-compatible base evaluation |
| TGVF Step 0 | base Qwen + RP66 Step 2000；尚未做 policy RL | PRL15-R1 paired snapshot，optimizer step 0 |
| Crop Step 8 | full-model Qwen；clean-final Crop RL | PRL14-A permanent Step 8 |
| TGVF Step 8 | full-model Qwen + trainable RP66 联合 RL | PRL15-R1 permanent Step 8 |

### 3.1 共同 base model

| 字段 | 值 |
|---|---|
| HF model | `Qwen/Qwen3-VL-8B-Instruct` |
| HF revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| local path | `/nvmesv/dredvpn009/models/hf/Qwen3-VL-8B-Instruct` |
| tokenizer length | `151669` |
| chat-template content SHA256 | `3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4` |
| weight dtype | BF16 |
| quantization | none |

### 3.2 Crop Step 0 的复用证据

PRL14 没有重复落盘一个 Step 0 目录，而是复用 PRL13 clean paired Step 0。该
复用是可审计的：

- Step 0 明确绑定 base HF tree，`optimizer_step=0`；
- base-equivalence proof SHA256：
  `6bad92612fc283b8c6974ea06c66c2d5d88288bfeadb4d7b6de229c791ba9bb9`；
- PRL13 Step 0 与 PRL14 Step 8 使用相同 Crop prompt bundle：
  `2b8b6d799ebe4bbfd6b3830344850575141b2293750f857c031a2031426c0dd2`；
- 两者 task manifest 均为：
  `3f69119d24867c3f3210c8b01eb71304247725ddaf9ca983d2b41c2885403cbc`；
- 均为 `deepeyes_official_visible_native_crop_v1`、原图 `0..1000` 坐标、最多
  6 次 Crop、plain final answer。

因此本文将 PRL13 clean Step 0 与 PRL14 clean-final Step 8 配对。PRL13 的历史
Step 8 训练仍带旧 `<answer>` 历史问题，不进入本基线。

### 3.3 RP66 Step 0 与 Step 8

RP66 初始化为：

```text
artifacts/representation/
  RP-66-qwen3-instruct-balanced-t1-contextual-2000-gpu01/adapter.pt
```

| 字段 | 值 |
|---|---|
| artifact storage SHA256 | `3429dc83880d48d623f3dcdbea48eb5219be6031b206e53c8286c4b1c65ce5c9` |
| artifact manifest SHA256 | `425d57108b60b2a2bc65144eaded0b1d25f763f3d347aa16396394dccae2b89c` |
| RP66 run identity SHA256 | `97ccfd849e1d66cdd57be805c27524fa97ca60973e5be45d6d060acd5bc54e53` |
| conditioning | last-layer contextual hidden state |
| Step 0 RP66 state SHA256 | `76cb3bea03076cb7763d63378edc6dcbdf05bade2b1b9793114b2f64eb0b28cd` |
| Step 8 RP66 state SHA256 | `8fbefbb51c9d537d8863413b1e90ef99c892592feea303bde2ad0c94b05dbecf` |

PRL15-R1 不是 policy LoRA。Qwen vision encoder、merger/projector、language model
以及 RP66 Adapter 都进入训练与每步 weight sync；正式评测的
`lora_request=null`，snapshot backend 为 `full_model_trainable_rp66`。

## 4. Policy-RL 训练配置

### 4.1 共同的科学变量

Crop Step 8 与 TGVF Step 8 被设计为匹配以下变量：

| 配置 | Crop PRL14 | TGVF PRL15-R1 |
|---|---:|---:|
| algorithm | GRPO | GRPO |
| optimizer updates used in comparison | 8 | 8 |
| prompt batch / update | 16 | 16 |
| rollouts / prompt | 16 | 16 |
| trajectories / update | 256 | 256 |
| total prompts through Step 8 | 128 | 128 |
| total trajectories through Step 8 | 2,048 | 2,048 |
| PPO epochs | 1 | 1 |
| optimizer | AdamW | AdamW |
| learning rate | `1e-6` | `1e-6` |
| scheduler | constant, no warmup | constant, no warmup |
| gradient clipping | 1.0 | 1.0 |
| GRPO advantage normalization | within-group std | within-group std |
| KL in reward | 0 | 0 |
| actor KL loss | off | off |
| entropy coefficient | 0 | 0 |
| rollout temperature / top-p | 1.0 / 1.0 | 1.0 / 1.0 |
| max prompt / response tokens | 8,192 / 20,480 | 8,192 / 20,480 |
| maximum tool calls | 6 | 6 |
| final answer | plain text；无 `<answer>` | plain text；无 `<answer>` |
| trainable Qwen path | vision + merger + LM | vision + merger + LM |
| extra trainable module | none | RP66 Adapter |

Crop 的完整 run 继续到了 Step 16，但本文只使用永久保留的 Step 8；TGVF 本次
正式 run 的目标和终点均为 Step 8。

### 4.2 分布式拓扑与 micro-batch

| 配置 | Crop PRL14 | TGVF PRL15-R1 |
|---|---:|---:|
| FSDP world size | 8 | 4 |
| FSDP strategy | FSDP2 | FSDP2 |
| prompt micro-batch / rank | 2 | 2 |
| prompt accumulation | 1 | 2 |
| nominal actor micro-batch / GPU | 32 trajectories | 32 trajectories |
| Step 8 effective actor micro-batch | 32 | 16 trajectories；4 local chunks / rank |
| global trajectories / update | 256 | 256 |

TGVF 使用 world4 是 world8 Crop 的数学等效 global-batch 实现，不是物理拓扑的
bitwise 等价复现。PRL15-R1 最后一次 update 的 actor micro32 因长序列 OOM，恢复
时改为 micro16 并增加本地 micro-batch 数；全局 256 trajectories、loss 归一化、
optimizer step 与 LR 不变。该差异必须保留为已知限制，后续不能把两条曲线称为
“完全相同执行拓扑”。

### 4.3 代码与运行身份

| 项目 | Crop PRL14 | TGVF PRL15-R1 |
|---|---|---|
| run name | `PRL-14-A-QWEN3-INSTRUCT-GRPO-BS16-N16-NATIVE-CROP-T1-CLEANFINAL-16STEP-WS8` | `PRL-15-R1-QWEN3-INSTRUCT-FULL-RP66-BS16-N16-CROP16-MATH-EQUIV-WS4` |
| run identity | completion contract `cb15857a...` | `16ac167482dcca0d6f8dfd4a9ae542d568a028c0a0237b13e31824767922d86d` |
| run config SHA256 | completion overrides | `843f368f00ed8e4d8ed0f948835b0457ea8c4640c6b1b2f884ef062a0b481616` |
| project launcher / final provenance | `2c7b0b014dbd867b1bfc2f809b8f15ecf7981abb` | `7324577ad1f8236cef49a73a0e67eb90aaf42861` |
| project source-state SHA256 | completion-bound | `93398e10f76797ed6531a9d1897739c97e88f678410b82246b3d181e9b90b1c9` |
| veRL commit | `e003163181731412595257a72ec173071efb125f` | same |

TGVF W&B：<https://wandb.ai/mio_nora/tgvf-policy-rl/runs/shx17t53>

PRL14 completion 中继承的内部 `run_id` 仍写成 PRL13，是历史元数据瑕疵。本文以
PRL14 `run_name`、artifact path、completion contract 和 snapshot hash 识别该
实验，不使用继承的内部 ID 命名它。

## 5. 数据与固定 schedule

两条 RL 线使用同一份修复 ThinkLite 类型后的最终 T1 数据：

```text
artifacts/data/policy_rl/
  T1-04-INSTRUCT-FULL-MIXED-T1-RETAINED-FINAL-v2/
```

T1 先用原始 `Qwen3-VL-8B-Instruct` 在 full-image、无工具条件下生成 8 个
rollout，保留天然答对次数为 `1..7/8` 的样本，排除 `0/8` 与 `8/8`。

| source | candidates | retained | retained ratio |
|---|---:|---:|---:|
| VStar | 170,000 | 39,205 | 23.06% |
| ArxivQA | 32,000 | 25,393 | 79.35% |
| ThinkLite | 69,842 | 12,943 | 18.53% |
| **total** | **271,842** | **77,541** | **28.52%** |

| identity | SHA256 / value |
|---|---|
| manifest file | `752ebe9ea5fced48773b9bc0babfbb6bc57a335dd1b580455f6962053d29fddf` |
| content | `5ab99622a2698a7c52c45795215fa5c467b741c103827a1a7dbe3800ff052934` |
| samples | `06e5b1b9039680111df5ef01f7f969b9cf3d8d0eaefa5774fd8d16169428611a` |
| schedule seed | `42` |
| shuffle | off |

共同的前 8-step schedule 消费 128 个唯一 prompt：VStar 56、ArxivQA 44、
ThinkLite 28。每个 prompt 生成 16 条 rollout。

## 6. 工具协议与 reward

### 6.1 Crop

```text
image_zoom_in_tool(bbox_2d, label?)
```

- `bbox_2d` 使用不可变原图上的 Qwen `0..1000` 相对坐标；
- observation 是真实 RGB crop，不是自然语言描述；
- 最多 6 次成功调用；
- prompt bundle：
  `2b8b6d799ebe4bbfd6b3830344850575141b2293750f857c031a2031426c0dd2`；
- system prompt SHA256：
  `1fc5b8b5ebdc9b24d6a9281071222872c8542dd65a4a4be1e70d9760c3a7f99f`；
- user suffix SHA256：
  `eac6399e048fc406c5a10fc44dd2f8d0c43c252e6f305b38844519ac71dbcfb0`。

### 6.2 TGVF

```text
tgvf_focus_tool(target)
```

- `target` 描述要提取的对象、区域、属性或关系；
- observation 是由原图视觉特征和 contextual target 生成的 main `D` 与
  D-DeepStack latent，不是生成好的答案文字；
- 最多 6 次成功调用；
- prompt SHA256：
  `e74bb5e1253af107ff27badfcfaca747b94574e19677d22cfe42b0b1c0ba5633`；
- tool schema SHA256：
  `f33f61d48bc4341f88077e90afca941819769b6209eb54893a9ed6b44856aba5`；
- agent loop SHA256：
  `f42a0cdcc3f1ac25b3e277428743d4888bf3e1951ed44e6f3c00dbb8072bc47e`。

两者共享 clean-final user 约定，但 tool/system bundle 因 action 与 observation
语义不同而必然不同。

### 6.3 Reward

视觉样本定义：

- `A=1`：最终答案正确，否则为 0；
- `F=0`：格式合法，`F=-1`：格式非法；
- `C=1`：至少有一次成功工具 observation，否则为 0。

Crop 与 TGVF 使用相同的视觉 reward：

$$
R_{visual}=0.8A+0.2F+1.2AC.
$$

ThinkLite 不开放视觉工具：

$$
R_{ThinkLite}=1.2A+0.4F.
$$

训练答案 judge 为 text-only `Qwen2.5-72B-Instruct`，通过 OpenRouter 固定
DeepInfra provider，concurrency 16。judge config SHA256 为：

```text
fff705c59408f4863244ff28df3443176e85de83147344df6a2350859c233021
```

没有加载 32B VLM judge，也没有视觉 grounding/aesthetic reward。`C` 只表示
工具调用执行成功，不证明返回的 Crop/D 对答案具有反事实增益。这一点是下一轮
TGVF reward 调整必须处理的核心问题。

## 7. CoreDev-2511 评测协议

| 字段 | 固定值 |
|---|---|
| task manifest | `CoreDev2511-official-visible-v1/tasks.jsonl` |
| task manifest SHA256 | `3f69119d24867c3f3210c8b01eb71304247725ddaf9ca983d2b41c2885403cbc` |
| official rows | 2,511 |
| tool runtime 实际支持 | 2,240 单图 |
| held multi-image | 271（BLINK 240；MMMU-Pro 31） |
| VLMEvalKit | `7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f` |
| semantic judge | local `Qwen2.5-72B-Instruct` |
| max model length | 32,768 |
| inference concurrency / GPU | 8 |
| final answer | plain text；无 `<answer>` |

Original 原生推理全部 2,511 条，包括 271 条多图。Crop/TGVF 只实际推理 2,240
条单图；多图在 official-size scoring view 中确定性计错。因此：

- **主表**使用 BLINK 180 与 MMMU-Pro 269 的共同单图支持集；
- full-420/full-300 zero-padding 只作为官方尺寸辅助表；
- Original 的共同支持分数通过相同 `sample_id` 精确 join 得到，不是重新采样。

解析失败口径：Original 旧 scorer 有 2 条、Crop Step 8 旧 scorer 有 7 条随机
fallback 标记；逐条核验这些样本的 `hit` 全为 0，改成当前
`deterministic_incorrect` 后分数不变。Crop Step 0 有 2 条、TGVF Step 0 有 3 条、
TGVF Step 8 有 4 条 deterministic parse failure。无 judge API/system failure 被
计入主表。

## 8. 主结果：共同单图支持集

所有分数单位为 `%`。

| benchmark | Original | Crop S0 | TGVF S0 | Crop S8 | TGVF S8 |
|---|---:|---:|---:|---:|---:|
| VStarBench Overall | 50.79 | 78.01 | 65.45 | 76.96 | 60.21 |
| HRBench4K `type=all\|accuracy` | 50.00 | 62.00 | 56.00 | 54.00 | 50.00 |
| BLINK single-image（180） | 65.56 | 57.22 | 62.22 | 60.00 | 56.11 |
| OCRBench v2 Chinese | 46.48 | 37.45 | 35.90 | 51.21 | 31.36 |
| MMMU-Pro-10c single-image（269） | 39.03 | 43.87 | 42.01 | 47.58 | 45.35 |
| MathVista MINI | 74.33 | 62.67 | 67.00 | 67.67 | 64.67 |
| MathVerse five-version macro | 50.60 | 54.80 | 48.40 | 54.00 | 50.00 |
| **7 项等权诊断均值** | **53.83** | **56.57** | **53.85** | **58.77** | **51.10** |

七项诊断均值是上述七行的简单等权平均，不是任何 benchmark 的官方总分。

## 9. 辅助结果

### 9.1 Official-size zero-padding

这两行不可用于评价 Original 与工具 arm 的多图能力差异，因为 Original 原生处理
多图，而工具 arm 将不支持的多图直接填零。

| benchmark | Original | Crop S0 | TGVF S0 | Crop S8 | TGVF S8 |
|---|---:|---:|---:|---:|---:|
| BLINK full 420 | 64.29 | 24.52 | 26.67 | 25.71 | 24.05 |
| MMMU-Pro full 300 | 40.33 | 39.33 | 37.67 | 42.67 | 40.67 |

### 9.2 MathVerse five-version

| version | Original | Crop S0 | TGVF S0 | Crop S8 | TGVF S8 |
|---|---:|---:|---:|---:|---:|
| Vision Dominant | 51 | 52 | 44 | 52 | 50 |
| Text Dominant | 69 | 69 | 64 | 61 | 63 |
| Text Lite | 53 | 58 | 52 | 58 | 53 |
| Vision Only | 28 | 49 | 36 | 49 | 44 |
| Vision Intensive | 52 | 46 | 46 | 50 | 40 |
| **macro** | **50.60** | **54.80** | **48.40** | **54.00** | **50.00** |

## 10. RL delta 与健康度

### 10.1 外部 benchmark delta

| benchmark | Crop S0 -> S8 | TGVF S0 -> S8 |
|---|---:|---:|
| VStarBench | -1.05 | -5.24 |
| HRBench4K | -8.00 | -6.00 |
| BLINK single-image | +2.78 | -6.11 |
| OCR Chinese | +13.76 | -4.54 |
| MMMU-Pro single-image | +3.72 | +3.35 |
| MathVista | +5.00 | -2.33 |
| MathVerse macro | -0.80 | +1.60 |
| **诊断均值** | **+2.20** | **-2.75** |

Crop 的提升并非所有 benchmark 单调上涨，但新增能力覆盖 OCR、MMMU、MathVista
与 BLINK。TGVF 只有 MMMU 与 MathVerse macro 上升，同时 VStar、HR、BLINK、
OCR 与 MathVista 下降，属于广泛退化。

### 10.2 TGVF 训练内指标与外部结果不一致

PRL15-R1 的训练日志表面上没有数值爆炸：

| 指标 | Step 1 | Step 8 |
|---|---:|---:|
| mean answer reward | 0.6094 | 0.5977 |
| mean conditional-tool reward | 0.4727 | 0.4961 |
| tool-call attempt rate | 75.00% | 81.25% |
| format error rate | 3.52% | 1.17% |
| gradient norm | finite | 2.50 at Step 8 |
| policy clip fraction | low | 1.34% at Step 8 |

8 step 共消费 2,048 trajectories，累计 judge cost 约 `$0.337`；记录的纯
end-to-end step 时间合计约 2.26 小时。训练 reward/格式看似稳定，但外部 CoreDev
显著下降，说明主要问题更像 **reward 与真实 TGVF 工具效用错位或 RP66 online
drift**，而不是简单的梯度发散。

### 10.3 参数漂移与 tool-success 饱和

基于 Step 0 / Step 8 冻结 snapshot 的只读 tensor 比较：

| parameter group | relative L2 delta | 备注 |
|---|---:|---|
| RP66，104 tensors | `5.254e-5` | cosine `0.9999999959`；max abs delta `1.526e-5` |
| full Qwen，8.767B params | `3.479e-5` | cosine `0.999999999395`；max abs delta `1.526e-5` |
| Qwen language | `3.768e-5` | — |
| Qwen vision encoder | `1.833e-5` | — |
| visual/deepstack merger | `8.754e-5` | 全局相对漂移最大的 Qwen 子系统 |

RP66 的 target projection 与 gate projection 分别约为 `1.73e-4` 与 `1.03e-4`。
RP66 总漂移按 step 单调累积，并非单步爆炸。这些数值不能单独证明退化原因，
但支持一个更具体的风险：RP66 是在 frozen Qwen reader 上完成 Stage 1 对齐的，
policy RL 同时移动 RP66、vision/merger 与 language reader 时，微小但协同的参数
变化也可能破坏原有 D-reader contract。

工具侧则接近饱和：2,048 条 trajectory 中共有约 1,629 次 TGVF 调用尝试、1,622
次成功 observation，执行成功率约 `99.57%`；累计有工具调用的 trajectory 比例为
`77.93%`。因此当前 reward 中的 `C` 几乎只区分“调用/不调用”，基本不能区分
“有用 D/无用 D”。

## 11. 后续 TGVF RL 的调整边界

本节记录下一轮实验设计边界，不把尚未执行的方案写成结果。

### 11.1 第一优先诊断：冻结 RP66

保持 PRL15-R1 的数据、128 prompts、seed、BS16、n16、8 step、LR、reward、
prompt 与 Qwen full-model 更新全部不变，只冻结 RP66 Adapter。

理由是 Crop 的 observation generator 固定，而 PRL15 同时训练了 policy 和
72M RP66。冻结 RP66 后，`Step 0 -> Step 8` 可以直接判断广泛退化是否来自
answer-level high-variance policy gradient 对已训练视觉表征的破坏。

这里的 “frozen RP66” 仍不是完全 immutable 的 TGVF 工具：共享 Qwen vision
encoder 仍会随 policy 更新。它是成本最低、变量最少的诊断；若要完全冻结
observation generator，需要维护独立 frozen vision+RP66 副本，会显著增加显存与
工程复杂度，不作为第一轮。

- 若冻结后恢复外部增益：下一步再研究 RP66 较小 LR、two-timescale update 或
  延迟解冻；
- 若冻结后仍退化：问题更可能位于 routing/target policy 或 reward。

### 11.2 第二优先单变量：降低 conditional tool reward

保持 RP66 可训练和其他配置不变，仅将：

```text
conditional_tool_weight: 1.2 -> 0.2
```

当前正确且成功调用工具的 reward 为 2.0，正确但不调用只有 0.8；然而 `C` 只证明
runtime 成功，不证明 `D` 对答案有用。该权重对可靠像素 Crop 有效，不代表可以
无条件迁移到 learned latent tool。

实现时不能只改 TOML：当前 DeepEyes source-aware resolver 可能应用 legacy visual
weights。smoke/formal 都必须记录并核对 runtime `applied_weights`，证明实际使用的
权重确实是 `0.8 / 0.2 / 0.2`。

### 11.3 暂不把 KL 作为首个调整

Crop 在相同 `KL=0` 下得到净提升；TGVF 的 clip fraction 与 gradient norm 也没有
显示 policy update 爆炸。首轮加入 KL 会同时增加新变量、计算和显存，且不能区分
RP66 drift 与 tool-reward 错位。KL 可以在上述两项定位后再评估。

### 11.4 后续结果的最低报告要求

任何声称优于本基线的 TGVF run 必须至少报告：

1. 完整 run/config/checkpoint/prompt/RP66 identity；
2. 相同 CoreDev manifest 下的 Step 0 与目标 checkpoint；
3. 本文七项 common-support 表及非官方诊断均值；
4. BLINK/MMMU 的 common-support 与 zero-padded 分数分开；
5. tool-call rate、成功 observation 数、answer/format/tool reward；
6. RP66 是否训练、独立 LR、参数漂移与 D-health 指标；
7. world size、实际 actor micro-batch、accumulation 与 OOM/restart 历史。

## 12. 已知限制

- 五个 arm 各只有一次正式 generation/scoring，没有多 seed 置信区间；
- Original 与工具 arm 的 prompt/agent protocol 不同，只能作为端到端参考；
- TGVF Step 0 含 RP66，不能用于估计“仅加入一个空工具”的效果；
- Crop 与 TGVF 的 observation 分别为真实 RGB 与 learned latent，这是 treatment
  本身而不是应消除的配置差异；
- Crop world8 与 TGVF world4/micro-accumulation 不是 bitwise 相同拓扑；
- TGVF 最后一个 update 使用 actor micro16，前面使用 nominal micro32；
- 工具 runtime 暂不支持多图，271 条任务只能在 common-support 之外单独解释；
- 训练 judge 是远端 text-only Qwen2.5-72B，正式评测 judge 是本地同模型；
- 当前结果能定位 RL 退化，但尚未给出 RP66 Step 8 的独立 foveation/D-health
  因果评测。

## 13. Artifact 与复现索引

### Original

```text
artifacts/evaluation/
  PRL-04-R2-raw-instruct-coredev2511-gpu4567-r4/
```

### Crop Step 0

```text
artifacts/evaluation/
  PRL13-A-CoreDev2511-clean-no-answer-paired-mem080-v1/
    step0/scoring/coredev-official-v2/coredev-2511-eval-summary.json
```

### Crop Step 8

```text
artifacts/policy/
  PRL-14-A-qwen3-instruct-grpo-bs16-n16-native-crop-t1-cleanfinal-16step-ws8/
    completion.json
    permanent-checkpoints/global_step_8/

artifacts/evaluation/
  PRL14-A-CoreDev2511-cleanfinal-step0-step8-step16-v1/
    step8/scoring/coredev-official-v2/
```

Crop Step 8 evaluation weights SHA256：
`54bf8864114b4b2b80c7603349d02425681a584fe7e4c6ea2c2b3d17fd4ae25d`。

### TGVF Step 0 / Step 8

```text
artifacts/policy/
  PRL-15-R1-qwen3-instruct-full-rp66-bs16-n16-crop16-math-equiv-ws4/
    launch-provenance.jsonl
    metrics.jsonl
    permanent-checkpoints/global_step_8/
    evaluation/
      PRL15-R1-RP66-COREDEV2511-STEP0-STEP8-SAME-PROTOCOL-RUNTIMEFIX-V2/
        paired-summary.json
```

| TGVF snapshot | weights SHA256 |
|---|---|
| Step 0 | `c7899330bc2419178a35a4a0f8837d47ee0361601a6954cb2717edbc6c39999b` |
| Step 8 | `c63a024939ae732c9dcbd9beb5ce90c688e73e5510f6319a0401e5e3a3859e63` |

冻结的 TGVF run config 位于实验 branch/worktree：

```text
configs/policy/runs/
  prl_15_r1_qwen3_instruct_full_rp66_bs16_n16_t1_crop16_math_equiv_8step_ws4.toml
```

其 file SHA256 为
`843f368f00ed8e4d8ed0f948835b0457ea8c4640c6b1b2f884ef062a0b481616`。
在不依赖 worktree 路径时，应从冻结 commit 读取：

```bash
git show 7324577ad1f8236cef49a73a0e67eb90aaf42861:\
configs/policy/runs/prl_15_r1_qwen3_instruct_full_rp66_bs16_n16_t1_crop16_math_equiv_8step_ws4.toml
```

paired scorer 的 deterministic/resumable 修复 commit 为
`e06450d9a5f30e1d213e64ff8cbf933ce1ccfe4c`。
