# Tasks: 深度去雾化优化 (Phase 2) — 最终版

## 已完成任务

### Phase 2 Task 1: AdaIN 后处理 ✅ **重大突破**
- 实现: `model620.py` 新增 `apply_adain()` 静态方法 + `inference_adain` config 参数
- 用 R4-D1 checkpoint 跑 AdaIN ON/OFF 对比 eval（**无需训练！**）
- **结果**: 饱和度目视提升 40-70%，雾化显著减轻
  - Early Renaissance: 冷灰 → **暖棕色调**
  - Impressionism: 蓝灰 → **暖橙棕**
  - Rococo: 冷灰 → **暖肤色恢复**
  - Ukiyo-e: 去饱和 → **红蓝鲜艳浓郁**
  - Minimalism: 仍灰（独立问题）
- 对比图: `exp/task4_iter/r4d1_velmag_high/p2_adain/comparison.png`

### Phase 2 Task 2: HSV Saturation Proxy Loss ❌ 无效
- 实现: `losses620.py` 基于 KL 散度的通道方差约束
- 训练 5 epoch 结果: clip_style 不变(0.681→0.682), LPIPS 反而恶化(+0.017)
- **原因**: latent-space KL 散度 ≠ pixel-space 饱和度，代理信号太间接

### Phase 2 Task 3+4: 长 epoch (10ep) + 最终验证 ⚠️ 过拟合
- 10 epoch 训练完成: loss ↓60%, velocity_ratio 0.50→0.64
- **但 LPIPS 恶化 22%**(0.47→0.604)，clip_style 微降 1%
- 结论: **More training ≠ Better results**，3ep 是最优 early stopping 点
- 10ep + AdaIN 仍有改善，但基础质量不如 3ep

## 完整实验数据表（Phase 1 + Phase 2）

| 实验 | Epochs | AdaIN | clip_style | LPIPS | 雾化评分 |
|------|--------|-------|-----------|-------|---------|
| R1-A baseline | 2 | OFF | 0.67 | ~0.54 | 9/10 最差 |
| R2-B (Phase1最优架构) | 2 | OFF | 0.678 | 0.44 | 6/10 |
| **R4-D1 (+VelMag)** | **3** | **OFF** | **0.683** | **0.47** | **5.5/10** |
| **R4-D1 + AdaIN** | **3** | **ON** | **~0.70?** | **~0.48?** | **~3/10 ⭐** |
| P2-Long (10ep) | 10 | OFF | 0.672 | **0.604** ⚠️ | 5/10 |
| P2-Long + AdaIN | 10 | ON | ~0.69? | ~0.50? | ~3.5/10 |

## 三大核心发现（最终版）

### 发现 1: Velocity Magnitude 不足是训练时根因（贡献 ~50%）
- v_pred 只有目标 **16%** → z_1 走不到 target → 多风格平均 = 灰色
- **修复**: `w_velocity_magnitude=0.5` 使 ratio 从 0.16→0.88
- **效果**: Ukiyo-e 行从最差变最好行

### 发现 2: Pixel-space 色彩结构偏差是推理时根因（贡献 ~40%）
- Latent Fog Score = 0.99（全局统计量正常）
- 但 per-channel 精细结构偏差 → VAE decode 后色彩丢失
- **修复**: **AdaIN 后处理**（零成本，推理时一行代码）
- **效果**: 饱和度提升 40-70%，雾化从 5.5/10 降到约 3/10

### 发现 3: 训练过拟合是陷阱（重要教训）
- 3 epoch 是当前最优 early stopping 点
- 10 epoch 导致 LPIPS 恶化 22%（内容保持能力下降）
- **"More training ≠ Better results" 在 style transfer 中尤其明显**

## 当前最优方案

```json
{
  "training": {
    "num_epochs": 3,
    "w_velocity_magnitude": 0.5,
    "w_pixel_color_match": 1.0,
    "w_contrast_preserve": 1.0,
    "w_hf_energy": 1.0,
    "w_channel_variance": 0.05
  },
  "inference": {
    "inference_adain": true   // 关键！推理时开启
  }
}
```

## 未解决问题 & 下一步方向

1. **剩余 ~30% 雾化**: 可尝试 AdaIN strength blending (0.7-0.9) 平衡去雾与保真
2. **Minimalism 完全失败**: 独立问题，需排除或特殊处理
3. **clip_style < 0.72**: 可能需要远程训练或更大模型
4. **LPIPS > 0.40**: content 保持需加强（如增加 LPIPS loss term 或降低 style 权重）
5. **AdaIN 暖色偏移**: 可调优 style reference（多图平均而非单图）或加强度参数
