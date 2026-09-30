# 06 — 代码清理与重构记录

> 历史清理记录 + 待执行的清理建议。用户指示"可以顺手清理 重构 优化"，本文档记录所有已完成和待执行的清理工作。

---

## 1. 历史清理记录（已完成）

### 1.1 628/629 清理

| 删除内容 | 理由 |
|----------|------|
| 9 项辅助 loss | 验证无效（Δclip=±0.0001） |
| `spectral_w_hh` | L8 确认 HH DEAD |
| `attn_modes`: gated/gated_raw/style_select/sparsemax | 验证无效 |
| FiLM modulation, style MoE, learnable shortcut | 验证无效 |
| skip_coarse, top-k truncation, style_bias | 验证无效 |
| 多级 DWT 分支 + Brownian 噪声分支 | active config 永不启用 |

### 1.2 630 Phase 1 清理

| 阶段 | Commit | 内容 |
|------|--------|------|
| Phase 1A | `925b6bea7` | H1-H11 零风险 dead code 删除（~80 行） |
| Phase 1B | `69da87cb0` | M9 attn_mode bug TDD 修复（relu2 传播） |
| Phase 1C | `bcea0a41b` | Legacy 文件批量删除（-11346 行，含 LANCET ~2070 行） |
| Phase 1D | `9de1e9e03` | 最简 codebase 性能验证（3-epoch PASS） |

### 1.3 630 Phase 6 (DINO 退役)

- 13 文件修改
- 所有功能性 DINO 引用从 `src/` 移除
- `style_memory` 成为唯一 style token 路径
- `dino_dim` 字段名保留仅为 checkpoint 兼容

### 1.4 T19a WCT 数值稳定性修复（本轮）

- **文件**: `src/spectral_bridge620.py::_wct_match_fiber`
- **修复**: 对角线正则化（`Σ + eps·I`）+ try-except 回退 AdaIN
- **原因**: depth=6 导致协方差矩阵病态，`eigh` 分解失败

---

## 2. 待执行的清理建议

以下 6 项清理建议在 [01_codebase.md](01_codebase.md) §5 识别，**待用户确认后执行**。

### 2.1 可删除的无效代码

#### 建议 1: T13/T14/T15 LLGSI/CASI/LLGQCA 代码

- **位置**: [src/blocks620.py](../../src/blocks620.py) L264-L318
- **状态**: T13-T16 系统性证明无效，方向已关闭
- **涉及 config 字段**:
  - `ll_global_style_inject: bool = False`
  - `ll_global_style_gate_init: float = 0.1`
  - `ll_style_inject_source: str = "style_mem"`
- **建议**: 直接删除代码 + config 字段
- **理由**: 用户偏好"无效代码确认后直接删除"

#### 建议 2: T5 eval_only_dwt_route 代码

- **位置**: [src/blocks620.py](../../src/blocks620.py) L189-L191
- **状态**: T5 失败，方向关闭
- **涉及 config 字段**:
  - `eval_only_dwt_route: bool = False`
- **建议**: 删除 `eval_only_dwt_route` 分支与 config 字段
- **理由**: 已确认失败

#### 建议 3: Phase 4J.6 endpoint style loss

- **位置**: [src/spectral_losses620.py](../../src/spectral_losses620.py) L79-L84
- **状态**: 4J.6 v3 验证无效（梯度通路太弱）
- **涉及 config 字段**:
  - `spectral_w_endpoint_style_lh: float = 0.0`
  - `spectral_w_endpoint_style_hl: float = 0.0`
- **建议**: 删除字段 + loss 计算代码
- **理由**: 已确认无效

#### 建议 4: wct_aligned_target (4J.2)

- **位置**: [src/spectral_losses620.py](../../src/spectral_losses620.py) L77-L78, L96-L110
- **状态**: 4J.2 方向未在 progress.json 记录为成功
- **建议**: **需确认**是否仍用于任何 active config
- **行动**: 扫描 configs/ 目录，若无 active config 使用则删除

### 2.2 可重构的复杂逻辑

#### 建议 5: integrate_transport 函数拆分

- **位置**: [src/spectral_bridge620.py](../../src/spectral_bridge620.py) `integrate_transport` 方法
- **问题**: ~250 行过长，含 Euler/Heun/RK4 + 4 种 schedule + 3 种 AdaIN 模式 + Progressive Alpha
- **建议**: 拆分为
  - `_euler_step(h, t, dt, ...)`
  - `_heun_step(h, t, dt, ...)`
  - `_rk4_step(h, t, dt, ...)`
  - `_apply_adain(h, style_latent, mode, alpha, ...)`
- **收益**: 可读性提升，单步测试更容易

#### 建议 6: SpatialBridgeBlock620 DWT route 分支抽取

- **位置**: [src/blocks620.py](../../src/blocks620.py) `forward` 方法
- **问题**: 3 种 mode (dwt_route / eval_only_dwt_route / dwt_route_train_prob) 嵌套
- **建议**: 抽取 `_compute_use_dwt()` 方法
- **收益**: forward 逻辑更清晰

---

## 3. 执行计划

### 3.1 立即可执行（低风险）

- **建议 1** (LLGSI/CASI/LLGQCA): 已绝对确认无效，可删除
- **建议 2** (eval_only_dwt_route): 已确认失败，可删除
- **建议 3** (endpoint style loss): 已确认无效，可删除

### 3.2 需确认后执行

- **建议 4** (wct_aligned_target): 需扫描 configs/ 确认无 active config 使用
- **建议 5** (integrate_transport 拆分): 重构，需测试验证
- **建议 6** (DWT route 分支抽取): 重构，需测试验证

### 3.3 验证方法

每项清理后执行：
1. **Smoke test**: 3-epoch 训练验证不破坏 baseline
2. **配置兼容性**: 确保 active configs 仍可加载
3. **Checkpoint 兼容性**: 确保 T11/4I.7b checkpoint 仍可加载

---

## 4. 清理决策矩阵

| 建议 | 风险 | 收益 | 用户偏好 | 建议执行？ |
|------|------|------|----------|-----------|
| 1 LLGSI/CASI/LLGQCA | 低 | 删除 ~55 行无效代码 | "确认后直接删除" | ✓ 是 |
| 2 eval_only_dwt_route | 低 | 删除 ~3 行无效代码 | "确认后直接删除" | ✓ 是 |
| 3 endpoint style loss | 低 | 删除 ~6 行无效代码 | "确认后直接删除" | ✓ 是 |
| 4 wct_aligned_target | 中 | 删除 ~14 行（待确认） | "确认后直接删除" | 需确认 |
| 5 integrate_transport 拆分 | 中 | 可读性提升 | 未明确 | 可选 |
| 6 DWT route 分支抽取 | 低 | 可读性提升 | 未明确 | 可选 |

---

## 5. 历史清理统计

| 阶段 | 删除行数 | Commit |
|------|----------|--------|
| 628/629 | ~3000+ | 多个 |
| Phase 1A | ~80 | `925b6bea7` |
| Phase 1C | -11346 | `bcea0a41b` |
| Phase 6 (DINO) | ~200 | 多个 |
| **总计** | **~14626+** | |

待执行清理预计删除: ~80 行（建议 1-3）+ 重构（建议 5-6 不删行）

---

## 6. 后续行动

1. **用户确认**建议 1-3 后立即执行删除
2. **扫描 configs/** 确认建议 4 是否有 active config 使用
3. **建议 5-6** 根据用户意愿决定是否重构（可推迟到论文提交后）

---

## 7. 清理执行日志

> 本节在执行清理时追加记录。

### 2026-07-02 (待执行)

- [ ] 建议 1: 删除 LLGSI/CASI/LLGQCA 代码
- [ ] 建议 2: 删除 eval_only_dwt_route 代码
- [ ] 建议 3: 删除 endpoint style loss 代码
- [ ] 建议 4: 扫描 configs/ 确认 wct_aligned_target 使用情况
- [ ] 建议 5: 拆分 integrate_transport（可选）
- [ ] 建议 6: 抽取 _compute_use_dwt()（可选）
