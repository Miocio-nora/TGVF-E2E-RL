# RP67 Adapter Ablation：计划、进度与结果

更新时间：2026-08-27 08:47 JST

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
| RP71 / no MatrixCE | **已完成 Step 2000 + internal eval** | same-image MatrixCE 权重 `1.0 → 0.0` | 主 `D` + 三路学习型 `D-DeepStack` |
| RP72 / no DeepStack | **已完成 Step 2000 + internal eval** | Adapter 变为 `main_d_only` | 仅主 `D`；三路分支固定为零且无分支可训练参数 |
| RP73 / unidirectional | **已完成 Step 2000 + internal eval** | 双向交互改为一次 target→visual payload 写入 | 主 `D` + 三路学习型 `D-DeepStack` |
| RP74 / post-merger | **已完成 Step 2000 + internal eval；Stage-1 优先候选** | Adapter 从 visual merger 前移到 merger 后 | 主 `D` + 三路学习型 `D-DeepStack` |

RP71 仍计算 raw MatrixCE 供诊断，但 weighted MatrixCE 严格为零，不进入总损失和梯度。
RP72 是结构/训练联合消融；它回答“学习型 DeepStack 分支是否必要”，不能与“仅在推理时
屏蔽 RP67 的 DeepStack”混为一谈。后者属于便宜的 deployment-use control，可在本轮
checkpoint 训练期间并行补。

RP73 的 active 路径只允许 target keys/values 写入 visual token。为了不把交互方向和
参数量混在一起，历史 visual→target enrichment 分支仍计算诊断 attention，但以严格零
系数接入输出；其参数梯度张量存在但数值为零。因此 RP73 与 RP67 都是 104 个 artifact
tensor、`72,055,808` 个可训练参数。

RP74 真正在 merger 输出之后交互：输入由 `[N,1152]` 变为 `[N/4,4096]`，四路输出直接
identity writeback，不再次调用 Qwen merger。attention bottleneck 仍固定为 `1152`，但
4096 维 visual projection、delta 和 gate 使参数量自然增至 `174,601,216`。这属于注入
位置结构消融的固有变化，结果解释不能假装它与 RP67 参数匹配。

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
- RP73/RP74 结构实现 commit：`ea14377c6e8837a43f38422c8756704da0b1c4e3`
- RP73 outer config canonical SHA256：
  `778eb764cd9abf410704ce56effdecf1ff8fa14b25bd3b3e64e740c3dbf3b111`
- RP74 outer config canonical SHA256：
  `8853131bce360d60aa32ef153e6cc574671a5e02722c367adf124b4089c4ea33`
- RP73/RP74 的 CPU/data/donor/identity fail-closed 预检均通过：8,209 个 donor assignment，
  其中 8,173 matched、36 masked；没有初始化 CUDA。
- 第一批相关 objective/config/image-axis 测试：`53 passed`；新增结构实现的完整
  representation 测试：`553 passed`。

## 5. 运行安排

| Arm | GPU | 预计训练时间 | 产物目录 |
|---|---:|---:|---|
| RP71 no MatrixCE | 0,1 | 已完成并释放 GPU | `artifacts/representation/RP-71-qwen3-instruct-rp67-ablation-no-matrixce-2000-gpu01/` |
| RP72 no DeepStack | 2,3 | 已完成并释放 GPU | `artifacts/representation/RP-72-qwen3-instruct-rp67-ablation-no-deepstack-2000-gpu23/` |
| RP73 unidirectional | 4,5 | 已完成并释放 GPU | `artifacts/representation/RP-73-qwen3-instruct-rp67-ablation-unidirectional-2000-gpu45/` |
| RP74 post-merger | 6,7 | 已完成并释放 GPU | `artifacts/representation/RP-74-qwen3-instruct-rp67-ablation-post-merger-2000-gpu67/` |

四臂均采用 2-GPU FSDP2。RP73/RP74 先确认 step 1 完成、loss/gradient 有限且参数
所有权检查通过，再转为正式 2,000-step 运行；后续 checkpoint 评测不能抢占训练 GPU。

### 5.1 启动验收

两臂于 `2026-08-26 21:38:50 JST` 并行启动。immutable start event、data identity、
split-overlap identity 与 run identity 均已落盘。`21:41 JST` 两臂都完成首个持久化
Step-10 metric，未出现 traceback、OOM、非有限 loss 或参数所有权错误。

| Arm | Step | raw MatrixCE | weighted MatrixCE | L_gen | Norm | weighted Norm | image-axis CE | Total | pre-clip grad norm | step wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RP71 no MatrixCE | 10 | 1.386686 | **0** | 5.035548 | 0.460832 | 0.046083 | 0.881284 | 5.962914 | 2.800789 | 8.004 s |
| RP72 no DeepStack | 10 | 1.384432 | 1.384432 | 4.910587 | 0.818586 | 0.081859 | 0.870416 | 7.247293 | 0.854934 | 7.544 s |

RP71 的 total 精确等于 `L_gen + 0.1 Norm + image-axis CE`，验证 raw MatrixCE 没有
进入优化目标。RP72 的 Adapter 参数面已由 `main_d_only` 配置内容绑定。当前 GPU 0–3
持续工作，GPU 4–7 空闲。训练主体按当前速度约 4.5 小时；最终 validation、checkpoint
导出和 post-training internal eval 需要额外时间。

### 5.2 S500 节点附近进度

`2026-08-26 22:46 JST`：RP71 到 Step 490，RP72 到 Step 520。RP72 的 S500 两 rank
checkpoint 已完整落盘；RP71 正在进入同一 checkpoint 边界。两臂均无 OOM、traceback、
非有限 loss 或进程退出。

| Arm | Step | raw MatrixCE | L_gen | image-axis CE | weighted Norm | Total | pre-clip grad norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| RP71 no MatrixCE | 490 | 1.368966（weighted=0） | 1.226086 | 0.006856 | 0.046898 | 1.279839 | 1.417634 |
| RP72 no DeepStack | 520 | 1.369618 | 1.636515 | 0.013054 | 0.047979 | 3.067166 | 1.827053 |

两臂 image-axis top-1 在该日志点均为 `1.0`。这只能说明训练目标当前可优化，不能在
internal/utility validation 完成前解释为下游质量。按启动至当前的端到端速度，训练主体
预计约 `2026-08-27 02:15–02:35 JST` 结束；自动 internal validation 后，再留约
20–30 分钟完成两臂 full-867 correct/zero/wrong utility，可在约 `03:00 JST` 前后形成
完整第一版结论。

### 5.3 RP73/RP74 结构与预检

`2026-08-26 23:18 JST`：两个新 variant 已实现并完成 CPU、artifact、checkpoint、
runtime、FSDP2、native pipeline 和 image-axis 全链路测试。RP73 outer config 的 source
SHA256 为 `941767ceb784e7814cfeb40f53a7110338d10c4f6b0d21ab2d33a948d11a6040`；
RP74 为 `e6115e5de6d958667bbbc9162c26fd38709827f3c7f2636d1a11af7a6e5348dc`。
两臂的 model、512² 图像上限、数据内容哈希、完整 objective、seed、optimizer、scheduler、
global batch 和 internal eval 与 RP67 对齐，并已通过 GPU 4–5 / 6–7 的启动验收与
Step-10 持久化检查。

两臂于 `2026-08-26 23:19:38 JST` 启动，并均已完成 Step-10 持久化指标：

| Arm | Step | MatrixCE | L_gen | image-axis CE | weighted Norm | Total | pre-clip grad norm | step wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RP73 unidirectional | 10 | 1.386515 | 5.029835 | 0.879372 | 0.046347 | 7.342069 | 1.921622 | 8.131 s |
| RP74 post-merger | 10 | 1.389620 | 4.978840 | 0.872325 | 0.018113 | 7.258899 | 11.588552 | 7.978 s |

两臂 image-axis top-1 均为 `0.21875`，loss、梯度和参数所有权均为有限值，无 OOM 或
traceback。RP74 的 pre-clip norm 较大，但已由共同的 `max_grad_norm=1.0` 执行裁剪；需要
继续观察其后续是否稳定。虽然 RP74 参数更多，但 merger 后 token 数为原来的四分之一，
Step-10 吞吐暂未慢于 RP73；这只是启动性能观察，不是质量结论。

### 5.4 2026-08-27 01:25 进度

四臂的 tmux、双 rank 训练进程和 GPU 占用均正常，无 OOM、traceback 或非有限值。
RP71/RP72 已跨过 S1500 checkpoint，RP73/RP74 已跨过 S500 checkpoint。

| Arm | Step | raw MatrixCE | L_gen | image-axis CE | weighted Norm | Total | pre-clip grad norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| RP71 no MatrixCE | 1680 | 1.335171（weighted=0） | 1.043074 | 0.000977 | 0.031952 | 1.076004 | 1.336213 |
| RP72 no DeepStack | 1780 | 1.328554 | 1.307445 | 0.007642 | 0.043075 | 2.686716 | 1.408716 |
| RP73 unidirectional | 910 | 0.922179 | 1.358059 | 0.003355 | 0.071165 | 2.354759 | 11.843270 |
| RP74 post-merger | 930 | 0.426972 | 1.076513 | 0.000199 | 0.061652 | 1.565335 | 4.951835 |

四臂该日志点的 image-axis top-1 都是 `1.0`。RP74 当前 MatrixCE 低于 RP73，但训练
日志是不同 minibatch 的单点观测，必须等待统一 validation 才能解释。按近期 wall time，
RP72 训练主体约 `01:55 JST` 完成，RP71 约 `02:10 JST`；RP73/RP74 训练主体约
`03:50–04:05 JST` 完成，随后各自自动运行 ordered-group、counterfactual 与 grounding
internal eval。

## 6. 完成结果

四个新 arm 均完成 2,000 step、最终 validation、Adapter 导出与同协议 internal eval，
GPU 0–7 已释放。下表只报告相同 200-sample/46-group 的表示内部协议；full-867
image+`D` correct/zero/wrong-target matched utility 尚未对 RP71–RP74 执行。

| Arm | val MatrixCE | val L_gen | val weighted Norm | val Total | retrieval top-1 | diag gap | MRR | correct > all controls | correct > wrong-same `D` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RP67 full/pre-merger | 0.201813 | 1.298340 | 0.072451 | 1.572604 | 0.900 | 3.0327 | 0.9463 | 0.945 | 0.950 |
| RP71 no MatrixCE | 1.386719 | 1.124512 | 0.029622 | 1.154133* | 0.515 | 0.0077 | 0.7068 | 0.725 | 0.725 |
| RP72 no DeepStack | 1.296875 | 1.543945 | 0.048275 | 2.889095 | 0.375 | -0.0142 | 0.6170 | 0.635 | 0.645 |
| RP73 unidirectional/pre-merger | 0.304688 | 1.226074 | 0.060507 | 1.591269 | 0.725 | 1.5823 | 0.8513 | 0.885 | 0.895 |
| **RP74 bidirectional/post-merger** | **0.021942** | **1.165527** | **0.049253** | **1.236723** | **0.935** | **6.2979** | **0.9667** | **0.965** | **0.965** |

`*` RP71 的 total 不包含 raw MatrixCE，不能用其较小 total 与其余 arm 直接比较。

RP74 相对 RP67 的 retrieval top-1 `+3.5 pp`、all-controls `+2.0 pp`、wrong-same
`+1.5 pp`，并把 mean diagonal gap 从 `3.0327` 提升到 `6.2979`。在 evaluator 报告的
31 个 grouped strata 中，RP74 在 retrieval top-1、all-controls 和 wrong-same 三项上均
无负 delta；其中 document (`n=31`) top-1 从 `0.8065` 到 `0.9355`。native
counterfactual continuation 从 `0.5000` 到 `0.6111`，expected-direction flip 从
`0.5556` 到 `0.6667`。

限制同样必须保留：RP74 的 native target-presence continuation 仍只有 `0.0417`，actual
direction accuracy 为 `0.1111`。因此 RP74 现在是 **Stage-1 优先候选**，还不是 RP67 的
正式下游替代。只有补完 full-867 correct/zero/wrong-target utility 以及必要的 policy-level
验证后，才能改 production/default binding。

RP74 Adapter file SHA-256 为
`005a5144df8b75ac3ac7822ed558e987cb08781f1cc11c0c2bf0fa77145829c7`，artifact
manifest SHA-256 为
`b7ef606711d983efedb39e4588e92498ded71a0101a715380b3af0ba0f816d16`；internal report
payload SHA-256 为
`ea6368333f2f1eb82aa09f3ac72adeb58faf065aada9886eba670ad6c5734c91`。RP74 也已写入
常规 `PROJECT_TASK.md` 和 `EXPERIMENT_LEDGER.md`，不再只存在于 workshop 专用报告。

## 7. 第二优先级候选

1. 冻结 RP67、仅在评测注入时屏蔽三路 `D-DeepStack`，隔离 inference contribution。
2. no Norm（`0.1 → 0`），判断历史范数约束是否必要。
3. target-free / shuffled-target control。已有 correct/zero/wrong-target utility 可先复用，
   不应在读完现有证据前重复训练。

优先完成 RP71–RP74 与现有 RP67/RP66 的统一评测，再决定是否扩张训练矩阵。
