# RP67 Adapter Ablation：计划、进度与结果

更新时间：2026-08-26 21:37 JST

## 1. 目标

本轮只回答 RP67 Stage-1 Adapter 内部组件是否必要，不改变 Qwen3-VL-8B-Instruct、
训练数据、图像上限、target conditioning、seed、优化器、学习率、scheduler、global batch
或 2-GPU FSDP2 几何。

RP67 完整基线为：

`balanced same-image MatrixCE + L_gen + 0.1 Norm + image-axis CE`，并输出主 `D`
与三路 `D-DeepStack`。正式 checkpoint 已完成 2,000 step，后续消融均与它比较。

## 2. 第一批严格矩阵

| Arm | 状态 | 相对 RP67 唯一主变量 | 训练输出 |
|---|---|---|---|
| RP67 full | 已有 | 无 | 主 `D` + 三路学习型 `D-DeepStack` |
| RP66 / no image-axis | 已有，可复用 | 去掉 image-axis CE | 主 `D` + 三路学习型 `D-DeepStack` |
| RP71 / no MatrixCE | CPU 预检通过，待启动 | same-image MatrixCE 权重 `1.0 → 0.0` | 主 `D` + 三路学习型 `D-DeepStack` |
| RP72 / no DeepStack | CPU 预检通过，待启动 | Adapter 变为 `main_d_only` | 仅主 `D`；三路分支固定为零且无分支可训练参数 |

RP71 仍计算 raw MatrixCE 供诊断，但 weighted MatrixCE 严格为零，不进入总损失和梯度。
RP72 是结构/训练联合消融；它回答“学习型 DeepStack 分支是否必要”，不能与“仅在推理时
屏蔽 RP67 的 DeepStack”混为一谈。后者属于便宜的 deployment-use control，可在本轮
checkpoint 训练期间并行补。

## 3. 固定条件

- Model：`Qwen3-VL-8B-Instruct`
- image max pixels：`262,144 = 512²`
- conditioning：最后层 contextual hidden state
- train / validation：与 RP67 完全相同的 retained T1 splits 与内容哈希
- same-image group size：`K=4`
- seed：`42`
- optimizer：AdamW，LR `1e-4`
- scheduler：2,000 step historical cosine，100-step warmup
- global training geometry：world size 2，GA 4，每 optimizer step 8 个矩阵、32 行
- image-axis donor：与 RP67 相同，8,209 assignments，其中 8,173 matched、36 masked
- checkpoint：每 500 step，最终 2,000 step
- post-training：同一套 ordered-group、counterfactual 与 audited-grounding internal eval

## 4. 配置与代码身份

- 实现 commit：`e8ff02d47af540c22c7ab660c6c1fdb5c5d0692c`
- RP71 outer config canonical SHA256：
  `9d42b38d1c49df98c99ad7ee92657e2002a2cf4bac7aec000fe499517c33ac34`
- RP72 outer config canonical SHA256：
  `e29e7815b4c20c9d2dfb1bf4036c4e37901b5a5032d8873f43116f4b78978530`
- 两臂 CPU/data/identity fail-closed 预检均通过；没有初始化 CUDA。
- 相关 objective/config/image-axis 测试：`53 passed`。

## 5. 运行安排

| Arm | GPU | 预计训练时间 | 产物目录 |
|---|---:|---:|---|
| RP71 no MatrixCE | 0,1 | 约 8–10 h | `artifacts/representation/RP-71-qwen3-instruct-rp67-ablation-no-matrixce-2000-gpu01/` |
| RP72 no DeepStack | 2,3 | 约 8–10 h | `artifacts/representation/RP-72-qwen3-instruct-rp67-ablation-no-deepstack-2000-gpu23/` |

GPU 4–7 暂不占用，留给 checkpoint 评测和 cheap controls。两臂应并行启动；先确认
step 1 完成、loss/gradient 有限且参数所有权检查通过，再把状态更新为正式运行中。

## 6. 结果表（待填）

| Arm | Step | MatrixCE | L_gen | Norm | image-axis CE | internal matched utility | zero-D utility | wrong-target utility | grounding |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RP67 full | 2000 | 待统一抄录 | 待统一抄录 | 待统一抄录 | 待统一抄录 | 待统一抄录 | 待统一抄录 | 待统一抄录 | 待统一抄录 |
| RP66 no image-axis | 2000 | 待统一抄录 | 待统一抄录 | 待统一抄录 | N/A | 待统一抄录 | 待统一抄录 | 待统一抄录 | 待统一抄录 |
| RP71 no MatrixCE | 2000 | 待运行 | 待运行 | 待运行 | 待运行 | 待评测 | 待评测 | 待评测 | 待评测 |
| RP72 no DeepStack | 2000 | 待运行 | 待运行 | 待运行 | 待运行 | 待评测 | 待评测 | 待评测 | 待评测 |

正式解释必须同时看表示内部指标和下游 matched utility。单独的训练 loss 下降不能证明
Adapter 更有效；RP72 参数量更小，也不能把吞吐变化误写成精度收益。

## 7. 第二优先级候选

1. 冻结 RP67、仅在评测注入时屏蔽三路 `D-DeepStack`，隔离 inference contribution。
2. no Norm（`0.1 → 0`），判断历史范数约束是否必要。
3. target-free / shuffled-target control。已有 correct/zero/wrong-target utility 可先复用，
   不应在读完现有证据前重复训练。

优先完成 RP71、RP72 与现有 RP67/RP66 的统一评测，再决定是否扩张训练矩阵。
