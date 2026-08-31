# 主代码统一与协议整合审计（2026-08-31）

本文档记录统一分支上已经进入真实 `config -> loader -> typed config -> veRL plan -> Hydra compose -> runtime/reward` 链路的能力。历史 worktree 中存在但未进入该链路的实现，不计为完成。

- 分支：`stabilize/protocol-contract-v1-20260830`
- 实现代码提交：`d1e8bbe658217371bbe571e3217c00ca94950043`
- 稀疏监督 native-pipeline 收口：`b8917f5`
- 五臂共享配置指纹：`f3bf6a38ba36efd66537c211c05853d3c64001e923bd9247c0ac7018ebb49392`
- 当前状态：主代码与默认配置入口已完成 CPU 收口；最终整合候选为
  `2381 passed, 4 skipped` 且 Ruff 通过。2026-09-01 用户明确暂缓 GPU
  训练与评测，因此没有 GPU/Ray 通过结论。

## 结论

本轮整合的核心结果不是“把 @512 写死”，而是建立了一个统一、显式、可审计的实验接口。当前 PRL28 选择 `262,144 = 512²` pixels、S32、BS16、`n=16`、seed 42、WS8 和 Teacher25，只是五臂 canonical 配置共同选定的一组值。分辨率、训练步数、checkpoint 节点、rollout 数、seed、batch/capacity、工具调用上限、reward 系数及开关、native DeepStack 和 Adapter 模式都由配置进入真实运行路径。

五个 RL treatment 是 NoTool、Crop、TGVF-short、TGVF-target-guide-v2 和 Atomic。Original Qwen 保留为 **eval-only comparator**，不伪装成第六份训练配置，也不能与 NoTool RL checkpoint 混为一谈。正式比较时，Original 必须与五个 treatment 共享 resolution、benchmark subset、scorer 和 evaluator provenance。

跨臂 validator 默认要求五臂齐全，除方法定义与输出位置等明确白名单外，对其余加载后配置递归比较并生成共享指纹。整组统一改成其他 resolution 或 horizon 是合法的新矩阵；只改一臂会在具体配置路径上报错。

### 正常启动入口

`run-policy <config.toml>` 直接加载普通实验配置、构建同一份 veRL plan、应用方法矩阵 overlay，并通过经过清理的子进程环境启动。它不要求 one-time token、runtime-locator manifest、freeze override 或把配置复制进另一个 canonical 目录。`dev-run-policy` 暂时保留为同一路径的兼容别名。

旧的 content-bound authorization 路径仅以 `strict-run-policy` 显式保留，供专门维护旧控制面时选择；它不是普通实验入口，也不进入默认 CI。

## 统一实验接口

新配置 schema 为 `policy-e2e-method-matrix-run-config-v1`。canonical 文件如下：

| Treatment | 工具能力与 Prompt | 成功 observation | 训练行为 |
|---|---|---|---|
| [NoTool](../configs/policy/runs/prl_28_a_qwen3_instruct_no_tool_pixel512_s32_bs16_n16_teacher25_ws8.toml) | direct-only；无工具 schema | no-tool/no-execution | 更新完整 Qwen；所有工具尝试零执行 |
| [Crop](../configs/policy/runs/prl_28_b_qwen3_instruct_crop_pixel512_s32_bs16_n16_teacher25_ws8.toml) | `image_zoom_in_tool`；DeepEyes matched prompt | Crop matched | 更新完整 Qwen，使用 exact crop replay |
| [TGVF-short](../configs/policy/runs/prl_28_c_qwen3_instruct_tgvf_short_pixel512_s32_bs16_n16_teacher25_ws8.toml) | `tgvf_focus_tool`；short matched prompt | TGVF matched | 更新完整 Qwen，RP67 Adapter 冻结并参与 exact replay/发布身份 |
| [TGVF-target-guide-v2](../configs/policy/runs/prl_28_d_qwen3_instruct_tgvf_target_guide_v2_pixel512_s32_bs16_n16_teacher25_ws8.toml) | 同一 TGVF 工具；只细化 Target 的视觉定义和 teacher-style 示例 | TGVF matched | 与 TGVF-short 相同，只把 Prompt 作为 treatment 差异 |
| [Atomic](../configs/policy/runs/prl_28_e_qwen3_instruct_atomic_pixel512_s32_bs16_n16_teacher25_ws8.toml) | `tgvf_crop_tool`；Atomic matched prompt | Atomic matched | 更新完整 Qwen，冻结 RP67；source/crop 双路 exact record/replay |

Target-guide-v2 没有额外修改 `<think>`、final-only、observation 文本、工具次数或 action boundary，因此它能用于单独判断“更详细 Target 定义”是否有效。

当前接口明确暴露并向下传递：

- `model.image_max_pixels` 与 `model.native_deepstack_enabled`；
- `scheduler.total_steps`、`training.maximum_optimizer_steps` 和 `training.checkpoint_steps`；
- `sampling.trajectories_per_prompt`、response length、采样参数和 rollout seed；
- global/micro batch、world size、FSDP/vLLM capacity；
- method、tool profile、matched observation、action boundary 和 `maximum_tool_calls`；
- answer/repeated-call/protocol reward 系数，以及 utility/visual-quality 开关；
- Adapter update mode、weight-sync mode、输出与恢复位置。

未来新增分辨率、步数或 ablation 时，应新增或派生一整组配置并运行矩阵 validator，不应在 launcher/runtime 中增加实验值分支。新增参数只有完成“解析、typed config、plan、compose/runtime、非默认值测试”后才算真正暴露。

## 已闭环的关键语义

### 1. T-free reward 是真实的无 utility/visual 链路

五臂 canonical reward 均关闭 tool-utility sidecar、visual judge、focus 和 grounding reward，也不读取 utility label，不做 answer gate。其有效总分为：

`R = answer_reward_scale * I(answer_correct) - repeated_call_penalty * max(tool_call_count - 1, 0) - protocol_error_penalty * I(any_protocol_error)`

当前 canonical 系数为 `2.0 / 0.05 / 2.0`，但它们是有限、非负的配置值，不是 kernel 常量。结果仍保留 answer/tool/focus/grounding/protocol 五组件 schema，T-free 下 focus/grounding 为 0，utility 与 judge audit 字段为 `None`，从而兼容现有 metrics 而不偷偷调用外部监督。历史 Stage3 utility-enabled 配置的语义未被改写。

### 2. matched observation 与 action boundary 统一

Crop、两个 TGVF 和 Atomic 分别绑定自己的 matched success-observation identity；所有方法绑定同一个 strict single-terminal action boundary v2。工具方法必须采样出完整 `</tool_call>`，下列输出在 parser/tool runtime 前即判为无效并保持零执行：

- close tag 后有非空 suffix；
- 同一轮出现多个完整 tool block；
- tag 畸形或不闭合；
- NoTool trajectory 尝试调用任何工具。

NoTool 没有 `</tool_call>` stop string。为避免 sampler 把协议伤害提前抹掉，它会保留模型实际采样出的 trailing/multiple tool output；随后由 direct-only action boundary 标记 `INVALID_FORMAT`，记录 `is_tool_call=true`，但不执行工具。这让“模型是否试图调用工具”和“工具是否执行”都可审计。该宽容只用于保留 NoTool 的无效样本，工具臂仍执行严格 close-tag 合约。

### 3. full-Qwen behavior identity 不再借用 LoRA pointer

NoTool/Crop 在上游完整 Qwen `update_weights` 成功返回后发布 typed behavior receipt；TGVF/Atomic 发布完整 Qwen receipt 加 RP67/RP66 Adapter state、storage 和 fan-out ACK 的组合身份。这样，rollout、metrics 和 checkpoint 引用的是本方法真实服务的行为，而不是不存在的 decoder-LoRA latest pointer。

这里必须保留一个边界：`accepted_full_qwen_sync_receipt_v1` 证明的是“某 run/step/request 的上游同步调用已经成功返回”，**不是 Qwen tensor content hash**。它足以闭合当前 transport acceptance 与行为版本，但不能被论文表述为逐 tensor 内容证明。

### 4. checkpoint sleep 后执行同 step resync

veRL level-2 rollout sleep 会丢弃已同步的完整模型权重。checkpoint wrapper 现在在保存成功后，对声明需要恢复权重的 manager 直接执行同一 optimizer step 的 `update_weights`，而不是只做 bare wake；同 step resync 保持稳定 PolicyVersion，同时记录新的 request/ACK。checkpoint 保存失败时会先唤醒 replicas 再重新抛出异常，避免留下睡眠 worker。

这修复了一个会让“checkpoint 后继续 rollout”使用无权重 replica 的 P0 问题。CPU 测试已覆盖同 step resync、禁止 bare wake 和失败恢复；真实 GPU sleep/wake 仍需 canary。

### 5. legacy metrics resume 不再猜测新字段

旧 checkpoint 没有 `maximum_tool_calls` 和 `trajectories_per_prompt`。恢复代码不再猜成历史默认 `4/8`，而是用 `None` 表示“旧文件未绑定”，序列化时继续省略这两个字段，从而保持旧 nested payload 及 project digest 原样。下一条经过当前配置验证的 observation 才把它们收敛为本次运行的真实 cap/n。

因此 cap=1 的 NoTool/Stage3 可以从旧 checkpoint 继续，而不会因为错误默认值被拒绝；新 checkpoint 仍严格绑定并验证真实 cap 和 group size。

### 6. native DeepStack 与 RP72 表征 ablation 独立

`model.native_deepstack_enabled` 从 TOML 同时进入 actor/reference HF override、vLLM HF override 和运行记录。关闭时，HF exact replay 不注入 native DeepStack；vLLM 仍保留主特征加三路 DeepStack 的既有 transport 形状，但在进入 language layer 前将 native injection 置零。

该开关控制的是 Qwen 原生 DeepStack，不等于 RP72 的 Adapter 表征设计（如 `main_d_only`）。两者可以独立组合，不能再用一个名称或布尔值替代另一个 ablation。

### 7. 当前 weight-sync interval 的诚实限制

`distributed.weight_sync_interval_optimizer_steps` 已是显式配置并进入 metadata，但 pinned synchronous veRL 当前没有 interval scheduler，因此只接受 `1`。其他正整数会被明确拒绝，而不是被静默忽略。若未来要稀疏同步，必须先实现并验证 engine scheduler，再放宽 schema；仅删除校验会制造假接口。

### 8. 旧公共兼容校验器不再制造 NoTool 假拒绝

旧校验器曾无条件要求 `limit_images >= 3`，把 NoTool 的“source image + 一次不可执行工具尝试”容量 `2` 错误拒绝。现在无 method binding 时只校验至少容纳 source image；传入 typed policy binding 时，再由 `1 + maximum_tool_calls` 做精确相等校验。这样既不放松已配置实验的容量检查，也不再把旧 Crop/TGVF 假设施加给 NoTool 或未来方法。

## 配置可变性证据

测试没有只用 @512/S32 默认值：

- loader/typed-config 使用 `image_max_pixels=345,678`、`n=3`、response 1,234、cap=5、seed=77、S2；
- 完整 `loader -> reward -> plan -> Hydra compose` 使用 pixels 456,789、`n=3`、cap=5、seed=91、S3，并验证 compose 后仍为这些值；
- 单一配置面传播测试使用 pixels 777,777、`n=5`、response 2,345、cap=7、seed=99、S7；
- 矩阵 validator 接受五臂统一从 262,144 改为 589,824，并生成不同共享指纹；只把 Atomic 改成 589,824 会准确报告 `model.image_max_pixels` 和 `policy.image_max_pixels` 漂移；
- validator 也会拒绝单臂 n、batch、S80、seed43、reward、dataset、agent-loop、capacity、precision 或 optimizer 漂移；
- native DeepStack 的 true/false 都覆盖了 actor 与 rollout 传播。

这组证据说明 @512/S32 是当前 canonical matrix 的 config selection，不是运行代码的隐藏锁。

## 测试快照

2026-09-01 源码晋升收口重新验证了当前 HEAD，而不是沿用旧文档数字：

| 范围 | 结果 |
|---|---:|
| 五臂 config/matrix、action/observation、exact replay、behavior/checkpoint focused core | 137 passed |
| 默认 Python 3.12 CPU suite | 2381 passed, 4 skipped |
| Ruff（`src tools spikes tests`） | passed |

在收缩默认测试面之前，全量历史 collection 实测为 2963 passed、4 skipped、57 failed。其中 56 个 failure 属于已经暂停的 token/freeze/runtime-locator/worker-bootstrap/control-plane 系统，另 1 个是专用 CUDA FlexAttention parity 在本机 Triton 编译环境下失败；这与“仅有 3 个历史 failure”的旧记录不一致，旧数字不再作为依据。

默认 pytest 现在明确排除这组已退休的 launch-security/control-plane 文件、repository-boundary CI ratchet 和专用 CUDA parity。它们仍保留在仓库中，可在对应维护任务里显式运行；普通 CI 不再执行 repository-boundary audit 或 launch-control/freeze audit。核心 config/matrix/action/replay/checkpoint tests 继续保留为默认门槛。

## 尚未完成与下一步

CPU 闭环足以完成源码晋升，但不等于新的训练结果可以直接宣称 golden。当前非阻塞后续为：

1. **GPU/Ray training canary（已准备、暂缓执行）**：Crop 与 Atomic 的
   @512 S1-to-S2 配置、checkpoint/resume materializer 和命令记录在
   `docs/PRL28_GPU_TRAINING_CANARY_20260901.md`；其状态仍是未运行。
2. **full-Qwen eval canary（尚未实现）**：需要非破坏的 Qwen-only merger
   与来源 receipt，机械证明 merged model 来自目标 checkpoint；Atomic 还需
   严格剥离 `tgvf_adapter.*`。未满足这些条件前不保留看似可运行的入口。
3. **完整 Qwen content proof（如论文需要）**：若要声称 tensor-level exact identity，需要在上游 transport 增加可验证的 tensor digest/manifest；当前 receipt 只能声称 accepted sync，不应过度解释。
4. **Original matched eval**：为未 RL 的 Original Qwen 建立独立 eval provenance，在相同 resolution、七个 subset、scorer、prompt edition 和统计代码下与五臂比较；它仍不进入训练矩阵。
GPU canary 是新 @512 S32 训练矩阵的运行验收，不是源码进入 `main` 的条件。通过后再把新的训练结果升级为可用于论文的 golden evidence。以后切换 1M、S16/S80、不同 n、reward 或 DeepStack/Adapter ablation 时，复用同一个 schema/launcher/runtime，通过 config 和矩阵指纹建立新实验，不再复制 PR 项目或新增专用分支入口。
