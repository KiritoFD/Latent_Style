# 616 探索+消融计划 — 16h 预算

> 运行中: `exp/20250618_lite_ot_vertical/` — b24, vl=0.1, legacy tokenizer + topogate
> 每 epoch ~30s, eval 每 4 epoch ~4min, 每个实验收敛 ~50min

---

## 阶段 1: 全因子探索 (7 实验, ~6h)

每个实验测试一条独立假说。共享基线配置: `legacy_factorized` + `ablation_disable_spatial_prior=true` + `semantic_self_topology_gate=true`。

| 实验 | bridge_path | coupling_cost_composition | coupling_structure_cost_mode | solver | sigma | 测试 |
|------|------------|---------------------------|------------------------------|--------|:---:|------|
| h0 | vertical | structure_only | self_affinity_gw | sinkhorn | 0 | 垂直 FM 基线 |
| h1 | linear | structure_only | self_affinity_gw | sinkhorn | 0 | 垂直 FM 效果 |
| h2 | vertical | appearance_only | — | sinkhorn | 0 | 欧氏 OT 对照 |
| h3 | vertical | structure_only | self_affinity_gw | sinkhorn | 0.02 | SDE 噪声 |
| h4 | vertical | structure_only | self_affinity_gw | sinkhorn_unbalanced | 0 | 非平衡 OT |
| h5 | vertical | appearance_plus_structure | topogate_attention_gw | sinkhorn | 0 | TopoGate attention complexity + latent self-affinity |
| h6 | vertical | appearance_plus_structure | topogate_attention_gw | sinkhorn_unbalanced | 0.02 | 全组合 |

**判据**: 每个实验跑完后读 clip_lpips_curve.csv。选出 best transfer style 和 best LPIPS。

---

## 阶段 2: 消融阶梯 (4 实验, ~4h)

从阶段 1 的全组合 (h6) 逐项剥除，量化每项贡献:

| 消融 | 基于 h6 移除 | 预期 style | 预期 LPIPS | 量化 |
|------|-------------|:---:|:---:|------|
| A1 | bridge_path_mode → linear | ↓ 0.01-0.02 | 不变 | 垂直 FM 的 style 贡献 |
| A2 | coupling_cost_composition → appearance_only | ↓ 0.01-0.02 | ↑ 0.01-0.02 | 结构 OT 的贡献 |
| A3 | bridge_sigma → 0 | ↓ 0.005-0.01 | 微降 | SDE 的贡献 |
| A4 | coupling_solver → sinkhorn | ↓ 0.005 | 不变 | 非平衡 OT 的贡献 |

**判据**: 找出贡献最大的模块，确认为核心机制。贡献 < 0.005 的模块标记为可选。

---

## 阶段 3: 最佳组合收敛 (1 实验, ~5h)

取阶段 1+2 中表现最好的配置，跑长训练 (120 epoch, b32, ~2h)。

**安全阈值**:
- LPIPS > 0.45 → 停止
- style 连续 6 eval 不增长 + LPIPS 不降 → 收敛, 停止
- ot_target_gini > 0.6 → OT 退化, 回退

---

## 时间预算

| 阶段 | 实验数 | 时间 | 可并行 |
|------|:---:|------|:---:|
| 1: 全因子 | 7 | ~6h | 否 (顺序) |
| 2: 消融 | 4 | ~4h | 否 |
| 3: 收敛 | 1 | ~5h | 否 |
| **总计** | **12** | **~15h** | — |

---

## gen_lite_batch.py 需要的改动

h5 / h6 的核心切换是:
- `coupling_structure_cost_mode`: `tokenizer_entropy_affinity_gw` → `topogate_attention_gw`
- `h5.coupling_cost_composition`: `structure_only` → `appearance_plus_structure`
- `h5/h6.coupling_structure_cost_weight`: 设到 `0.4`

代码依赖 `topogate_attention_gw` 模式需要在 `_structure_pairwise_cost` 中实现——
从 `model.last_semantic_topology_attn`（无则回退 `last_semantic_attn`）提取 attention 矩阵，
计算 entropy map，构建 self-affinity descriptor。
