# Policy RL CoreDev-2511 统一测量标准与主基线

日期：2026-08-12

状态：`PRIMARY MEASUREMENT CONTRACT / FROZEN V1`

Contract ID：`POLICY-RL-COREDEV2511-MEASUREMENT-20260812-v1`

适用范围：后续 Qwen3-VL-8B-Instruct Crop / TGVF policy-RL 的 Step 0、训练中间点与最终 checkpoint 对比。

本文冻结两件事：

1. CoreDev-2511 的统一测量与聚合口径；
2. 截至 2026-08-12 晚上的 canonical 大表。

本文只替代旧文档中的 headline 聚合值，不否定旧文档记录的模型、prompt、checkpoint、训练配置与 artifact 身份。特别是 `docs/POLICY_RL_PRIMARY_BASELINE_20260810.md` 中使用 HRBench cycle 0 和 OCR Chinese-only 得到的旧均值，不再作为主汇报值。

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
| 6 | MathVista MINI | `Task&Skill=Overall|acc` |
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

当前 content-addressed RNG 把 `evaluation_id` 纳入了 `trajectory_id`。因此即使权重、prompt 与 task 完全相同，只要换 evaluation ID，就会换采样随机流。

这不是可以忽略的理论问题：RP67 R1 与 R2 的 Step 0 权重完全相同，但 Macro* 分别为 `57.38` 和 `56.37`，观测差为 `1.01 pp`。

从下一组新正式对比开始，优先使用共享的 paired RNG namespace：

```text
seed = H(master_seed, task_manifest_sha, protocol_sha,
         sample_id, rollout_index, assistant_turn_index)
```

seed 身份必须排除 `evaluation_id`、arm 名、optimizer step 和 checkpoint hash，使同一题在 Step 0 / Step 8 / Step 16 使用同一随机流。该功能在实现并通过 CPU/GPU smoke 前不得宣称已经生效。

在 paired RNG 正式落地前，继续使用现有 `temperature=1` 协议，但小于或约等于 `1.01 pp` 的单次变化一律只写作“趋势”，不得写作“已确认提升/退化”。

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

## 3. Canonical 大表（2026-08-12）

所有数字单位为 `%`。OCR mean 是 EN/CN 的均值；Macro* 只把 OCR mean 计入一次。

| benchmark | Original | Crop clean S0 | Crop clean-final S8 | RP67 +T S0 | RP67 +T S8 | RP67 T-free S0 | RP67 T-free S8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| VStarBench Overall | 50.79 | 78.01 | 76.96 | 66.49 | 58.64 | 64.92 | 65.45 |
| HRBench Average / all | 59.00 | 53.50 | 62.50 | 59.50 | 60.00 | 58.00 | 62.50 |
| BLINK single-image（180） | 65.56 | 57.22 | 60.00 | 59.44 | 63.89 | 63.33 | 64.44 |
| OCRBench v2 English | 49.89 | 40.46 | 47.39 | 46.12 | 44.99 | 45.47 | 44.54 |
| OCRBench v2 Chinese | 46.48 | 37.45 | 51.21 | 34.19 | 37.66 | 36.35 | 37.83 |
| OCR EN/CN mean | 48.19 | 38.96 | 49.30 | 40.16 | 41.33 | 40.91 | 41.19 |
| MMMU-Pro single-image（269） | 39.03 | 43.87 | 47.58 | 48.33 | 47.58 | 45.35 | 48.33 |
| MathVista MINI | 74.33 | 62.67 | 67.67 | 73.33 | 65.67 | 68.67 | 65.67 |
| MathVerse five-version macro | 50.60 | 54.80 | 54.00 | 54.40 | 57.00 | 53.40 | 53.40 |
| **Macro\*** | **55.36** | **55.57** | **59.72** | **57.38** | **56.30** | **56.37** | **57.28** |

对应的 RL delta：

| 线路 | Step 0 | Step 8 | Delta | 当前结论 |
|---|---:|---:|---:|---|
| Crop clean-final | 55.57 | 59.72 | **+4.14 pp** | 大于已观察采样波动；当前最可靠的正向 pilot |
| RP67 +T | 57.38 | 56.30 | -1.08 pp | 单次负向趋势；幅度接近随机波动，不能单独定性 |
| RP67 T-free | 56.37 | 57.28 | +0.91 pp | 单次正向趋势；小于相同 Step 0 的 1.01 pp 跨评测波动，尚未确认有效 |

由此不能声称 RP67 T-free 已经有效。它目前只是最值得继续验证的 RP67 reward 线路。

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

### Crop clean-final Step 8

```text
artifacts/evaluation/
  PRL14-A-CoreDev2511-cleanfinal-step0-step8-step16-v1/
    step8/scoring/coredev-official-v2/
```

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
3. 数据 manifest、BS、rollout n、world size、micro-batch、GA、LR 和 reward 分解；
4. 本文九行分项表与 canonical Macro*；
5. normal final、direct answer、call cap、invalid format、max/context tokens 的数量和比例；
6. 平均/中位 response tokens、tool calls、成功 observation 数；
7. judge parse failure、judge/API/system failure 分开报告；
8. 同协议 Step 0 与目标 checkpoint；使用的 paired RNG namespace 或其尚未启用的明确声明；
9. 小于当前 `1.01 pp` 单次波动参照的变化，只能作为趋势。

## 6. RP67 T-free Step 8 → Step 16 验证计划

继续训练是合理的，因为它能区分三种情况：

- Step 16 继续高于 Step 8：支持正向 scaling 趋势；
- Step 16 回落到 Step 0 附近或更低：Step 8 的 `+0.91 pp` 更可能是采样波动或短暂峰值；
- Step 16 仍在约 `±1 pp` 内：结论仍是不确定，需要 paired seed 或重复 seed，而不是继续靠单次大表定性。

续训必须满足：

- 从现有 Step 8 的完整 model、optimizer、scheduler、data cursor 和 RNG state 原位恢复；
- 保持同一 run ID 与科学 identity；不能新建一份伪装成独立训练的 16-step TOML；
- RP67 Adapter 继续 frozen；full Qwen（包括视觉路径）继续更新；
- reward 继续为 T-free：answer + protocol + repeated-call penalty；tool utility、focus、grounding 继续关闭；
- 保持 BS16 prompts × n16、world8、LR `1e-6`、constant scheduler；
- 永久保留 Step 8 与 Step 16，rolling checkpoint 每步更新；
- Step 16 后重新评测 Step 0、Step 8 与 Step 16，三者使用同一 `temperature=1` paired seed namespace；不能把旧 Step 0/8 与一个新随机流的 Step 16 直接当作精确差值。

当前 Step 8 checkpoint 的 optimizer/data/RNG 状态完整。实测前八步纯训练共 `4,924.7 s`，因此 Step 8 → 16 预计训练 `87–95 min`，保守 `1 h 30 min–1 h 45 min`。只做 Step 8/16 两臂的完整 CoreDev temp=1 评测约 `52 min`；推荐的 Step 0/8/16 三臂配对评测约 `75–90 min`，训练加推荐评测合计约 `2 h 45 min–3 h 15 min`。

现有 launcher、supervisor 和 paired evaluator 把目标/arm 写死为 Step 8，不能直接启动。实现时应从实际训练提交 `d4286ca` 建立窄 continuation 分支，只加入 Step 8 → 16 continuation manifest、world8 恢复 gate、Step 16 永久保存和 Step 0/8/16 自动配对评测；不要把 temp=0 evaluator 的共享 sampler 改动带入训练代码。
