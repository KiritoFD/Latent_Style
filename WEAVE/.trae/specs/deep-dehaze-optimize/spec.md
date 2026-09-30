# 深度去雾化优化 Spec (Phase 2)

## Why

Phase 1（12轮迭代）已完成根因诊断并取得 **~40% 雾化改善**（9/10 → 5.5-6/10），但：
- Velocity ratio 从 16% → 88%（接近目标），但**雾化仍有 ~50% 残留**
- Latent Fog Score = 0.99 说明问题在 **pixel-space 精细色彩结构**，非全局统计量
- clip_style 停留在 0.68 左右，LPIPS 在 0.47+，距离实用水平有差距
- **需要从"治标"转向"治本"：直接在 pixel-space 约束色彩分布**

## What Changes

### 核心：Pixel-Space 色彩约束 + 推理后处理

1. **HSV Saturation Loss** — 直接惩罚生成图像的 S 通道低于 target
2. **AdaIN 后处理** — 推理时对 z_1_hat 做 adaptive instance normalization（零成本，无需重训练）
3. **长训练验证** — 用最优配置跑 10 epoch，观察 velocity_ratio 能否突破 0.95
4. **排除 Minimalism** — 临时移除该风格确认是否拖累整体
5. **修复 eval 图片生成** — 确保 summary_grid.png 稳定输出

## Impact
- Affected code: `src/losses620.py`（新增 HSV loss）、推理代码（AdaIN 后处理）、eval pipeline
- 继承: `anti-whiten-no-dino` Phase 1 的所有发现和最优配置基线
- 目标指标: clip_style > 0.72, LPIPS < 0.40, 雾化评分 < 4/10

## ADDED Requirements

### Requirement: HSV Saturation Preservation Loss

系统 SHALL 提供基于 HSV 色彩空间的饱和度保持损失，在 pixel level 直接约束生成图像的色彩饱和度不低于目标。

#### Scenario: Saturation Loss 生效
- **WHEN** `w_hsv_saturation > 0`
- **THEN** 将 z_1_hat 和 y_proj 解码为 RGB pixel，转换为 HSV 空间
- **THEN** 计算 S 通道均值比 `gen_sat_mean / tgt_sat_mean`
- **WHEN** 该比值 < threshold（默认 0.8）时产生梯度惩罚
- **THEN** 默认权重 0（向后兼容）

#### Scenario: VAE Decoder 可用性
- **WHEN** VAE decoder 未传入 loss compute 函数
- **THEN** 回退到 latent-space 的近似 saturation 估计（通道方差熵作为代理）

### Requirement: AdaIN Post-processing Inference

系统 SHALL 支持推理时可选的 Adaptive Instance Normalization 后处理步骤，将 target latent 的 channel-wise 统计量迁移到 generated latent 上。

#### Scenario: AdaIN 后处理启用
- **WHEN** `inference_adain = true`
- **THEN** 对 z_1_hat 执行: `output = s_tgt * (z - mean(z)) / std(z) + mean(tgt)`
- **WHERE** s_tgt, mean(tgt), std(tgt) 来自 projected_target (y_proj)
- **THEN** 不影响模型参数或训练流程（纯推理时操作）

### Requirement: Stable Eval Image Output

系统 SHALL 在每次 full eval 时稳定生成 summary_grid.png，包含 source/target/generated 三列对比图。

#### Scenario: Full eval 完成
- **WHEN** full evaluation 执行完毕
- **THEN** 自动生成 summary_grid.png 并保存到 eval 输出目录
- **THEN** 图片包含每个 style pair 的可视化对比
- **THEN** 无需手动配置开关（默认开启）

## MODIFIED Requirements

### 训练配置扩展
- 新增 `w_hsv_saturation: float = 0.0` 和 `hsv_sat_threshold: float = 0.8`
- 新增 `inference_adain: bool = false`
- 新增 `exclude_styles: list = []`（支持排除特定风格如 Minimalism）

## Open Questions
- [ ] HSV Saturation Loss 是否需要 VAE decode（增加计算开销）还是可以用 latent 近似？
- [ ] AdaIN 后处理是否会破坏已经学到的风格特征？
- [ ] 排除 Minimalism 后其他风格的指标能提升多少？
- [ ] 10 epoch 训练后 velocity_ratio 是否能突破 0.95？
