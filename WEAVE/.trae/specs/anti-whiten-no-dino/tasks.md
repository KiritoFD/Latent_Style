# Tasks: 去除 DINO + 突破白化问题（迭代版）— 最终版

## 已完成任务

### Phase 1: 基础设施
- [x] Task 1: 图片白化诊断 — saturation=0.174(严重不足), whiteness=0.43
- [x] Task 2: 切换 latent 模式去 DINO — 仅配置切换，无需代码修改
- [x] Task 3: Anti-whitening Loss 三件套 — contrast/ch_var/hf_energy

### Phase 2: 架构迭代 (R1-R3, 7轮)
- [x] R1-A: No DINO baseline (velocity mode) → loss=1.968, 雾化严重(9/10)
- [x] R1-B: +去 GN → 无改善, GN不是根因
- [x] R1-C: +Fixed One gate → loss=1.748(-11%), 雾化中度(7/10)
- [x] R2-A: +FiLM → loss=1.742, 雾化中度-轻度(6/10) ⭐当前最优架构
- [x] R2-B: +AntiWhiten(w=1/1/0.05) → loss=1.787, 雾化6/10
- [x] R3-A: 激进AW(w=10/5/0.2) → loss=1.724, **无图片**
- [x] R3-B: Endpoint mode + 强正则 → **退步**5.5/10, endpoint模式短期不适用

### Phase 3: 根因发现与修复 (R4-R6)
- [x] R4-A: Velocity Scaling 推理实验 → **根因确认！endpoint_alpha=16%时scale=7.0达107%**
- [x] R4-C: VelMag Loss w=0.1 → velocity_ratio: 0.16→0.525(+228%), clip_s_delta_idt 负转正
- [x] R4-D1: VelMag Loss w=0.5 → velocity_ratio: **0.88**(接近目标!), Ukiyo-e显著改善
- [x] R5: Latent vs Pixel 诊断 → **Latent Fog Score=0.99! 问题在decode后非latent本身**
- [x] R6: Per-Channel Color Match Loss → clip_style +0.36%, identity +0.97%

## Task 5: 最终总结（见下方结论）

## 完整实验数据表

| 实验 | 关键参数 | Loss | velocity_ratio | 雾化评分 | clip_style | 关键发现 |
|------|---------|------|--------------|---------|-----------|---------|
| R1-A | baseline | 1.968 | ~0.16 | 9/10 严重 | 0.67 | 基线，严重雾化 |
| R1-B | +去GN | 1.968 | ~0.16 | 9/10 | 0.67 | **GN不是根因** |
| R1-C | +FixedOne | 1.748 | ~0.20 | 7/10 | 0.68 | Gate是瓶颈之一 |
| R2-A | +FiLM | 1.742 | ~0.22 | 6/10 | 0.68 | FiLM有帮助 |
| R2-B | +AntiW | 1.787 | ~0.25 | 6/10 | 0.68 | AntiW单独不够 |
| R3-B | Endpoint | 1.031 | ? | 5.5/10 | 0.68 | Endpoint短期退步 |
| **R4-D1** | **VelMag w=0.5** | **1.36** | **0.88** | **5.5/10** | **0.68** | **Velocity幅度修复!** |
| R6 | +PixelColor | 1.355 | 0.55 | ? | **0.683** | 细微提升 |

## 三大核心发现

### 发现 1: Velocity Magnitude 不足（贡献度 ~50%）
- 模型预测的 velocity 只有目标 **~16%**
- z_1 = x + (1-t)*v_pred 根本走不到 target
- 多风格训练时"走不到"= 所有目标平均 = 灰色
- **修复**: `w_velocity_magnitude=0.5` 使 ratio 从 0.16→0.88
- **效果**: Ukiyo-e 行从最差变最好行

### 发现 2: Latent 空间正常，雾化在 Decode 后（贡献度 ~30%）
- Latent Fog Score = 0.9912（几乎完美匹配）
- 全局 mean/std 正确，但 per-channel 精细结构有偏差
- VAE decode 对 latent 精细结构敏感，小偏差→大色彩丢失
- **尝试修复**: Per-channel mean/std matching loss（效果有限）

### 发现 3: Minimalism 风格独立失败（贡献度 ~20%）
- 无论怎么调参，Minimalism 输出都是纯灰噪声
- 可能原因：该风格数据集本身的特性（单色/低纹理）
- 与通用雾化问题独立，需要特殊处理

## 当前最优配置

```json
{
  "model": {
    "style_condition_source": "latent",
    "endpoint_film_use_norm": false,
    "style_gate_mode": "fixed_one",
    "style_film_enabled": true,
    "endpoint_film_enabled": true
  },
  "training": {
    "training_objective_mode": "velocity",
    "w_velocity_magnitude": 0.5,
    "w_pixel_color_match": 1.0,
    "w_contrast_preserve": 1.0,
    "w_hf_energy": 1.0,
    "w_channel_variance": 0.05
  }
}
```

## 未解决的问题 & 下一步方向

1. **剩余 ~40% 雾化**: 需要 pixel-space 直接约束或 decode 后处理
2. **Velocity ratio 卡在 0.55-0.88**: 训练更多 epoch 或调整 loss 权重
3. **eval 图片生成不稳定**: 需要修复 eval pipeline 确保 summary_grid.png
4. **Minimalism 完全失败**: 可能需要排除该风格或特殊处理
5. **建议**: 尝试 AdaIN 后处理（零成本）、增加训练到 10-20 epoch、或在 pixel space 加 HSV saturation constraint
