# FC-SB Phase 3 深入排查与开关修复 Spec

## Why

**直接触发**：用户指出 R/T/U/V/W 方向评估结果与 baseline 完全相同，质疑"开关没生效"。诊断已确认：

1. **T/U/V 方向**：checkpoint config 正确（`endpoint_adain_scale=1.0, multiband_adain_mode='two_level'` 等），但推理时 `style_latent` 永远为 None，整个 N1 endpoint AdaIN 块（[model620.py:676](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L676)）被跳过。T1-T4、U1-U3、V1-V3 的 LPIPS 全部为 0.4180-0.4181，不随参数变化。
2. **W 方向**：W2b 训练时（12:49:54 启动）用的是旧 json（`w_anti_input_style=0.3`），新 json（3.0）在 12:50:51 才更新——训练启动早于配置更新。且 0.3 相对 `w_endpoint_style=8.0` 太小，2 epoch 内不可见。train.log 中无任何 W loss 输出。
3. **R 方向**：唯一生效的方向（独立于 N1），但 LPIPS 随 cfg 增大而恶化（0.42→0.43→0.46），是真实的"生效但恶化"。

**深层问题**：之前 spec（`fc-sb-phase3-directions`）将"代码已写"等同于"功能已生效"，缺乏 **运行时 probe 验证**。runtime_observability 早已显示 `model_endpoint_style_high_abs=0.0, model_style_dino_active=0.0`（style 路径完全未激活），但未被识别为"开关失效"信号。

**目标**：深入排查 T/U/V/W 四方向的"代码已写但运行时未走到"根因，修复后用 smoke test 验证开关真正生效，再决定是否进入参数搜索阶段。

## What Changes

### 工程约束（硬性）
- **显存控制**：smoke test 与正式评估显存 ≤ 11GB（RTX 3060 12GB）
- **算力复用**：修复时复用现有 checkpoint，不重新训练（除非确认配置错误）
- **probe-first 原则**：任何修复后，**先**用 probe 验证开关生效（runtime_observability 非零），**再**做完整评估
- **配置同步**：修复涉及的代码改动必须同步到远程 I 盘，并验证文件 mtime

### Change 1: 修复 T/U/V 推理时 style_latent 为 None 的根因

**根因定位**：
- [run_evaluation.py:3190-3196](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/utils/run_evaluation.py#L3190) 构造 `target_style_latent` 为 dict：`{"style_dino_cls": ..., "style_dino_patches": ..., "style_text_tokens": ...}`
- [model620.py:567-568](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L567) 判断 `if style_latent is None and target_style_latent is not None and not isinstance(target_style_latent, dict):` —— 由于 target_style_latent 是 dict，条件为 False，style_latent 保持 None
- [model620.py:676](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L676) `if endpoint_adain_scale > 0.0 and style_latent is not None and isinstance(style_latent, torch.Tensor):` —— style_latent 是 None，整个 N1 块（含 T/U/V）跳过

**理论问题**：N1 endpoint AdaIN 需要一个 **style_latent tensor**（用于提取 per-channel fiber 统计 μ/σ），但当前推理流程中 style 信息通过 dino_patches/cls 传递，没有真正的 "style_latent tensor"。需要明确：
- 训练时 N1 的 `style_latent` 从哪来？（查 trainer.py 调用链）
- 推理时应该从哪构造这个 tensor？（选项：DINO patches 投影 / 参考图 VAE encode / 其他）
- T/U/V 代码是否在训练时也走了同样的死路径？（如果是，则 T/U/V 在训练时也未生效）

**修复策略**（待实现阶段确定）：
- **方案 A**：在推理时从 target DINO patches 通过 style_conditioner 投影出 style_latent tensor
- **方案 B**：用 target 风格参考图的 VAE encode 作为 style_latent tensor
- **方案 C**：修改 L567 的判断逻辑，从 dict 中提取或构造 style_latent
- **方案 D**：若训练时也走死路径，则 T/U/V 在训练阶段也需要修复

### Change 2: 修复 W 方向训练配置加载时机

**根因定位**：
- W2b 训练启动时间：12:49:54（train.log L23）
- W2b.json 更新时间：12:50:51（文件 mtime）
- 训练启动早于配置更新 57 秒，用的是旧 json（w_anti_input_style=0.3）
- checkpoint 保存的 config 确认：`w_anti_input_style=0.3`（非 3.0）

**修复策略**：
- 删除 W2b 旧 checkpoint，用新 json（3.0）重新训练
- 训练脚本启动前**必须**验证 json 内容与预期一致（启动前 cat json | grep w_anti_input_style）
- 训练日志首行打印关键 weight 值，便于事后核查

### Change 3: 新增 probe-first 验证流程

**问题**：之前评估后发现"结果与 baseline 相同"才意识到开关失效，浪费算力。

**修复**：在 `run_rtuv_eval.py` 和 `run_w_batch.py` 中，训练/评估完成后**立即**检查 `runtime_observability.model_endpoint_style_high_abs`：
- T/U/V: 期望 `endpoint_style_high_abs > 0`（N1 块执行）
- W: 期望 train.log 中出现 W loss 分量（`anti_input_loss`, `fiber_repulsion_loss`, `style_disc_loss`）
- 若 probe 失败，标记结果为 INVALID，不进入后续分析

### Change 4: 反思与流程改进

**反思点**：
1. "代码已写" ≠ "功能已生效" —— 必须有运行时 probe 验证
2. `target_style_latent` 变量名误导（实际是 dict，非 tensor）—— 应重命名或文档化
3. runtime_observability 的 `model_endpoint_style_high_abs=0.0` 应作为 CI gate，非零才允许进入参数搜索
4. 训练配置加载时机：json 更新与训练启动之间有竞态，需要启动前校验

**流程改进**：
- 任何新方向的代码改动，**必须**先做单样本 smoke test，打印 runtime_observability，确认开关生效
- 评估脚本首行打印 config 关键字段，便于核对
- 训练脚本首行打印 loss 分量列表，确认所有 weight 非零的 loss 都在

## Impact

- **Affected specs**: `fc-sb-phase3-directions`（前置 spec，本 spec 修复其实现 bug）
- **Affected code**:
  - `src/model620.py` — L567-568 style_latent 赋值逻辑，L676 N1 块前置条件
  - `src/utils/run_evaluation.py` — L3190 target_style_latent 构造，L3209 generation_with_target_latent 调用
  - `src/trainer.py` — 训练时 style_latent 传递链（待排查）
  - `exp/625_fc_sb/run_rtuv_eval.py` — 新增 probe 检查
  - `exp/625_fc_sb/run_w_batch.py` — 新增 config 启动前校验 + loss 分量打印
- **Affected docs**: 无（修复后更新 EXPERIMENT_LOG.md 记录真实结果）

## ADDED Requirements

### Requirement: T/U/V 推理时 style_latent 必须非 None
系统 SHALL 在推理时为 N1 endpoint AdaIN 块提供非 None 的 `style_latent` tensor。
- **WHEN** `endpoint_adain_scale > 0.0` 且推理调用 `i2sb_inference`
- **THEN** `style_latent` MUST 是 torch.Tensor（非 None、非 dict）
- **验收**: runtime_observability.model_endpoint_style_high_abs > 0.0

### Requirement: W 训练 config 启动前校验
系统 SHALL 在 W 方向训练启动前校验 json 配置与预期一致。
- **WHEN** 启动 W 方向训练
- **THEN** 训练脚本 MUST 打印 json 中的 w_fiber_repulsion/w_anti_input_style/w_style_disc 值
- **THEN** 训练日志首行 MUST 包含所有非零 weight 的 loss 分量名
- **验收**: train.log 首行包含 `w_anti_input_style=3.0`（或预期值）

### Requirement: Probe-first 验证 gate
系统 SHALL 在评估/训练完成后立即检查 probe 指标，失败则标记 INVALID。
- **WHEN** R/T/U/V 评估完成
- **THEN** MUST 检查 `runtime_observability.model_endpoint_style_high_abs`
- **WHEN** W 训练完成
- **THEN** MUST 检查 train.log 中是否出现 W loss 分量
- **验收**: probe 失败的结果标记为 INVALID，不进入分析

## MODIFIED Requirements

### Requirement: 三阶段评估协议（增加 probe gate）
原 `fc-sb-phase3-directions` spec 的三阶段协议（参数搜索 → 最佳点训练 → 最终评估）增加前置 probe gate：
- **阶段 0（新增）**: smoke test 验证开关生效（单样本，打印 runtime_observability）
- 阶段 0 通过后才进入阶段 A（参数搜索）
- 阶段 0 失败则修复代码，不浪费算力跑无效实验

## REMOVED Requirements
无
