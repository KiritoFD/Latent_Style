# FC-SB 全量实验计划 — 一天时间预算

## Summary

基于 [FC.md](docs/622/FC.md) 文档的全部改造方案，结合 FC-SB v1 的实验结果（clip_style=0.704, LPIPS=0.559），制定一天内可完成的**全量实验矩阵 + 自主探索计划**。

## Current State Analysis

### 已完成的代码改造（3 文件，~60 行）
| 改造 | 文件 | 状态 |
|------|------|------|
| Base Locking + Fiber SDE | model620.py L531-594 | ✅ 已实现并验证 |
| Gate init 0.05→0.5 | blocks620.py L83 | ✅ 已实现 |
| 训练期高通 SDE 噪声 | losses620.py L354-362 | ✅ 已实现 |
| RMSNorm (body level) | blocks620.py L11-28 | ✅ E4 遗留，已启用 |

### FC-SB v1 实验结果（10 epoch, batch=12, ~58 分钟）

| 指标 | 值 | 判定 |
|------|-----|------|
| clip_style | **0.704** 🎉 | +4.8% vs E4, 风格提升有效 |
| LPIPS | **0.559** ⚠️ | 未达 <0.40 目标，但优于 E9(0.544)/E12(0.602) |
| velocity_std | 0.937 | 健康 |
| style_gate | 0.462 | 接近目标 0.5 ✅ |
| fiber_energy | 0.646 | 稳定但未突破 1.0 |

### FC-SB v1 配置中的关键参数（已激活）
```
transport_prediction_mode = "endpoint"     ← 可能是 LPIPS 差的元凶！
i2sb_fiber_project_endpoint = true        ✅
i2sb_fiber_project_noise = true           ✅
bridge_path_mode = "vertical"              ✅
bridge_sigma = 0.08                       ← 可能过高
training_target_projection_mode = "pure_vertical_flow_wavelet"  ✅
w_style_energy_floor = 0.5               ✅
body_norm_type = "rms_norm"               ✅
style_gate_init = 0.5                     ✅
loss_type = "mse"                         ← FC.md 建议 huber!
objective_mode = "flow_matching"         ← FC.md 建议 i2sb_endpoint!
```

### FC.md 提到但尚未尝试的改造（3 个高优先级）

| # | 改造 | 当前状态 | 影响 |
|---|------|---------|------|
| A | **Huber Loss** 替代 MSE | ❌ 未实现 | SDE 噪声产生离群点，Huber 更抗噪 |
| B | **objective_mode: "i2sb_endpoint"** | ❌ 当前是 flow_matching | 不同 Loss 计算路径 |
| C | **bridge_sigma 扫描** (0.02~0.08) | 仅试了 0.08 | 0.08 可能过高导致 LPIPS 恶化 |
| D | **回退 velocity 模式** | 当前 endpoint | XPred 模式可能不适合 Base Locking |
| E | **更多 epoch (15-20)** | 仅试了 10ep | 三阶段课程需要更多时间收敛 |
| F | **lowpass kernel 变体** (7 或 wavelet) | 仅用 k=5 avg_pool | 可能切割过于激进 |

---

## Proposed Changes — 实验矩阵（7 个实验，串行执行）

### 实验矩阵总览

| # | 实验名 | 核心变量 | 目的 | 预计时间 | 优先级 |
|---|--------|---------|------|---------|-------|
| E1 | **fc_sb_huber** | loss_type="huber", delta=1.0 | 抗 SDE 离群点 | 40min | P0 最高 |
| E2 | **fc_sb_sigma04** | bridge_sigma=0.04 | 降低噪声强度修复 LPIPS | 40min | P0 最高 |
| E3 | **fc_sb_velocity** | transport_prediction_mode="velocity" | 回归速度预测模式 | 40min | P0 最高 |
| E4 | **fc_sb_15ep** | num_epochs=15 | 更多时间让三阶段收敛 | 60min | P1 高 |
| E5 | **fc_sb_combo** | huber+sigma=0.04+velocity | 组合最优参数 | 50min | P1 中 |
| E6 | **fc_sb_kernel7** | fiber_kernel=7 | 温和频率切割 | 40min | P2 低 |
| E7 | **自主探索** | 基于 E1-E5 最佳结果动态调整 | 冲击帕累托 | 60min | P1 高 |

**总预计时间**: ~5 小时训练 + 1 小时 eval/diag = **6 小时**（留 18 小时 buffer 给排障和额外探索）

---

### E1: fc_sb_huber — Huber Loss 抗噪

**改动**: `src/losses620.py` compute() 方法

将 FM Loss 从 `F.mse_loss` 改为 `F.huber_loss`：
```python
# 原来: fm = F.mse_loss(pred_velocity.float(), target_velocity.float())
# 改为:
fm = F.smooth_l1_loss(pred_velocity.float(), target_velocity.float())
# 或 Huber: F.huber_loss(pred, target, delta=1.0, reduction='mean')
```

**配置**: 基于 FC-SB v1，仅改 `bridge.loss_type = "huber"`，其余不变。

**预期**: SDE 噪声产生的离群点不再主导梯度，LPIPS 应下降 5-10%

---

### E2: fc_sb_sigma04 — 降低 SDE 强度

**无代码改动**，仅配置变更：

```json
"bridge": { "bridge_sigma": 0.04 }  // 从 0.08 降到 0.04
```

**预期**: 减少高频噪声注入量 → LPIPS 显著下降（主要优化方向），clip_style 可能略降

---

### E3: fc_sb_velocity — 回退速度预测模式

**无代码改动**，仅配置变更：

```json
"model": { "transport_prediction_mode": "velocity" }
// 注意: 这意味着模型输出 v 而非 x1
// integrate_transport() 会自动适配（内部仍做 endpoint 转换）
```

**预期**: 速度预测模式更保守 → LPIPS 大幅改善（可能回到 0.35-0.45），clip_style 可能降到 0.68-0.70

---

### E4: fc_sb_15ep — 更长训练

**无代码改动**，仅配置变更：

```json
"training": { "num_epochs": 15 }
```

**预期**: 三阶段课程有更多时间收敛 → 两个指标同步改善

---

### E5: fc_sb_combo — 最优组合

基于 E1-E3 的结果，选取最佳参数组合。可能的组合：
- 如果 E2(sigma=0.04) 效果最好: huber + sigma=0.04 + velocity
- 如果 E3(velocity) 效果最好: huber + velocity + sigma=0.04
- 如果 E1(huber) 效果最好: huber + sigma=0.04 + endpoint

---

### E6: fc_sb_kernel7 — 温和频率切割

**代码微调**: model620.py 中 `fiber_kernel` 默认值从 5 改为 7（或通过配置传入）

```json
"model": { "i2sb_fiber_project_kernel": 7 }
```

**预期**: k=7 低通滤波保留更多中频信息 → 结构保持更好

---

### E7: 自主探索

根据 E1-E6 结果，自主决定下一步：
- 如果某实验接近目标（如 clip>0.69 且 LPIPS<0.42）→ 微调该方向
- 如果所有实验都确认了某个规律 → 总结规律并更新 Dashboard
- 可选: 尝试 FC.md 中提到的 **改造3 (Fiber Endpoint Prediction)** — 让网络只预测 Δf

---

## Assumptions & Decisions

1. **batch_size=12** 全局保持（FC-SB v1 证明安全，VRAM 3.43GB）
2. **每个实验独立部署训练**，不共享 checkpoint（从随机初始化或 E4 resume）
3. **串行执行**（单 GPU 无法并行）
4. **每个实验后运行 full_eval** 并提取 clip_style + LPIPS
5. **Dashboard 在全部实验完成后统一更新一次**
6. **Huber Loss 使用 SmoothL1Loss 实现**（PyTorch 原生支持，delta 默认 1.0）
7. **远程 GPU 空闲可用**（需验证）

## Verification Steps

### 每个 experiment 完成后检查:
1. [ ] 训练无 NaN / OOM / 崩溃
2. [ ] full_eval 完成，summary.json 存在
3. [ ] 提取 clip_style 和 LPIPS
4. [ ] 与 FC-SB v1 基线对比: clip_delta?, lpips_delta?

### 最终成功标准（按优先级排序）:
- [ ] **银牌**: 任一实验达到 clip_style > 0.69 且 LPIPS < 0.45
- [ ] **金牌**: 任一实验达到 clip_style > 0.71 且 LPIPS < 0.38
- [ ] **钻石**: clip_style > 0.73 且 LPIPS < 0.30（FC-SB 终极目标）
- [ ] Dashboard 更新全部新数据点
- [ ] 最终结论报告写入 spec checklist

## 时间预算表

| 时段 | 任务 | 累计 |
|------|------|------|
| T+0h | Huber Loss 代码修改 + E1/E2/E3 配置生成 | 0.5h |
| T+0.5h | **E1: fc_sb_huber** 训练 (10ep) | 1.5h |
| T+2h | **E2: fc_sb_sigma04** 训练 (10ep) | 3h |
| T+3.5h | **E3: fc_sb_velocity** 训练 (10ep) | 4.5h |
| T+5h | 分析 E1-E3 → 决定 E4/E5 参数 | 5h |
| T+5.5h | **E4 或 E5** 最优组合训练 (10-15ep) | 7h |
| T+7.5h | **E6** (如时间允许) | 8h |
| T+8h | 全部 Eval + Dashboard 更新 | 9h |
| T+9h | 自主探索 E7 (基于结果动态决策) | 10-12h |
| Buffer | 排障 / 重跑 / 额外消融 | ≤24h |
