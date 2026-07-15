# Checklist: 去除 DINO + 突破白化问题

## Phase 1: 基础设施
- [x] Task 1: 白化诊断基线 — saturation=0.174, whiteness=0.43, Rococo/EarlyRen 最严重
- [x] Task 2: Latent 模式切换 — 无需代码修改，验证通过
- [x] Task 3: Anti-whitening Loss 三件套 — contrast/ch_var/hf_energy, 默认0向后兼容

## Phase 2: 架构迭代 (R1-R3)
- [x] R1-A: No DINO baseline → 雾化严重(9/10)
- [x] R1-B: +去 GN → 无改善
- [x] R1-C: +Fixed One → loss -11%, 雾化7/10
- [x] R2-A: +FiLM → 雾化6/10 ⭐最优架构
- [x] R2-B: +AntiWhiten → 雾化6/10
- [x] R3-A: 激进AW权重 → loss改善但无图
- [x] R3-B: Endpoint mode → 退步5.5/10

## Phase 3: 根因发现与修复 (R4-R6)
- [x] R4-A: Velocity Scaling → **根因确认！scale=7时alpha=107%**
- [x] R4-C: VelMag Loss w=0.1 → ratio 0.16→0.525(+228%)
- [x] R4-D1: VelMag Loss w=0.5 → **ratio=0.88, Ukiyo-e显著改善**
- [x] R5: Latent vs Pixel诊断 → **Fog Score=0.99! 问题在decode后**
- [x] R6: Per-Channel Color Match → clip_style+0.36%

## 最终状态总结

### 已解决 ✅
- [x] DINO 去除：latent 模式完全可用，节省显存和时间
- [x] Velocity 幅度不足：从16%提升到88%，Ukiyo-e行大幅改善
- [x] clip_s_delta_idt 负转正：从-0.145到+0.048
- [x] 向后兼容：所有新参数默认0，不影响现有配置

### 部分解决 ⚠️
- [~] 整体雾化：从9/10降到约5.5-6/10（~40%改善）
- [~] 速度比：接近但未达到1.0（0.88 vs 目标1.0）

### 未解决 ❌
- [ ] 剩余雾化（pixel-space 精细色彩结构）
- [ ] Minimalism 风格完全失败（独立问题）
- [ ] eval summary_grid.png 不稳定生成

### 关键修改文件
| 文件 | 改动 |
|------|------|
| `src/config_schema.py` | +12个新参数（vel_mag, pixel_color, anti_whiten等）|
| `src/losses620.py` | +velocity_magnitude_loss, +velocity_direction_loss, +pixel_color_match_loss, +anti_whitening_3件套 |
| `src/model620.py` | +velocity_scale 推理参数 |
