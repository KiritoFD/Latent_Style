# 616 实验全景（收敛驱动版）

> 全局 `virtual_length_multiplier=0.1` → ~2.5 min/epoch
> 不固定 epoch, 收敛为止 → 连续 3 eval 的 style delta < 0.002 即停
> 每 4 epoch eval 一次, 最少 12 epoch (~30min), 最多 60 epoch (~2.5h)
> **当前运行**: `exp/20250618_lite_ot_vertical/` — b24, legacy_factorized + ablation_disable_spatial_prior=true + topogate

## 7 个实验: 6 个独立假说 + 1 个全组合

| 代号 | 假说 | 变量 | 基线 |
|------|------|------|------|
| H0 | 垂直 FM 有效 | `bridge_path_mode="vertical"` | — |
| H1 | 线性 FM 效果差 | `bridge_path_mode="linear"` | H0 |
| H2 | 欧氏 OT 不如结构 OT | `coupling_cost_composition="appearance_only"` | H0 |
| H3 | SDE 噪声突破均值 | `bridge_sigma=0.02` | H0 |
| H4 | 非平衡 OT 改善匹配 | `coupling_solver="sinkhorn_unbalanced"` | H0 |
| H5 | TopoGate attention > 其他结构代价 | `coupling_structure_cost_mode="topogate_attention_gw"` | H0 |
| H6 | 全组合最优 | H0+H2+H3+H4+H5 叠加 | — |

## 实验脚本

| 脚本 | 用途 |
|------|------|
| `tools/experiments/run_phase616_converge_ladder.sh` | **主脚本**: 7 个实验顺序跑 |
| `tools/experiments/run_phase616_vram_probe.sh` | VRAM 探测: 每种 tokenizer 的最大 batch |

## 运行

```bash
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
bash tools/experiments/run_phase616_converge_ladder.sh
```

预计总时间: ~10h (7 实验 × ~1.5h)

## OT 实现深度分析

### 匹配流程

```
1. _ot_match_targets()
   └─ 对每个 style_id 分组:
       └─ _solve_group_coupling()
          ├─ _coupling_cost_matrix()      ← 计算 N×N 代价矩阵
          │  ├─ transport_cost.pairwise_cost()  ← 外观代价 (欧氏/SWD)
          │  └─ _structure_pairwise_cost()      ← 结构代价 (GW affinity)
          ├─ _sinkhorn_plan()              ← 解 OT (标准 or 非平衡)
          └─ _sample_or_project_from_plan() ← 从传输计划采样目标
```

### 代价矩阵构成

最终 $C_{ij}$ = 归一化后的 外观 + 结构的加权和。当前默认 `structure_only` (只用结构, 权重=1.0)。

### 结构代价的 5 种描述符

| 模式 | 描述符维度 | 来源 | 状态 |
|------|:---:|------|------|
| `topogate_attention_gw` | 4+ | TopoGate attention 熵画像 | ✅ 推荐 — 零成本内生 |
| `self_affinity_gw` | 28 | 潜变量的 self-attention triu | ✅ 保留 baseline |
| `encoder_self_affinity_gw` | 28 | UNet encoder 特征的 self-attention | ⚠️ 额外 forward |
| `lowedge_self_affinity_gw` | 34 | 低频+边缘 + affinity blend | 📋 保留 |
| ~~`tokenizer_entropy_affinity_gw`~~ | — | Tokenizer 路由熵 + affinity | ❌ 废弃 (tokenizer 无 ROI) |

### 非平衡 OT

`coupling_solver="sinkhorn_unbalanced"` + `sinkhorn_unbalanced_tau_src=0.5`:
- tau_src 小 → 源侧放松 → 允许源图放弃不合适的匹配
- 等效于加一个"Dummy class"吸收坏配对
- 监控: `ot_source_truncation > 0` 表示哑类在工作

### 关键技术细节

- OT 匹配在 `torch.no_grad()` 下执行 → 不计梯度, 只影响目标选择
- `_ot_tokenizer_aux_feature_map()` 提取 tokenizer 的 attention 中间层 (无梯度)
- Gini 指标自动记录在训练 CSV 的 `ot_target_gini` 列
