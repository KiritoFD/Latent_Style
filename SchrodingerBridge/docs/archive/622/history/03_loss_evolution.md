# Loss函数演变史

## 总览

| 阶段 | 时间 | 主Loss | 辅Loss | 删除的Loss | 总权重数 |
|------|------|--------|--------|-----------|---------|
| SA-Flow | 01月 | FM-MSE | - | - | 1 |
| LGT-X | 01-28 | SWD+MSE | color, identity | - | 4 |
| C-G-W | 02月 | SWD | color, TV | cycle(replaced by MSE) | 4 |
| Cycle-NCE | 03月 | SWD | color(latent_decoupled_adain), identity, PatchNCE | structure | 5 |
| SB初始 | 05-07 | L_flow | L_terminal_swd, L_color, L_repulsive | - | 4 |
| SB kitchen-sink | 05-08 | L_flow | +L_kinetic, +L_nce, +L_cycle, +L_low_freq, +L_semantic_swd | - | 9 |
| SB清理后 | 05-19 | L_flow | L_kinetic, L_terminal_swd, L_curvature(可选) | color,repulsive,nce,cycle,low_freq,OMF | 3-4 |
| Distinct5 | 06月 | L_flow | L_kinetic, L_terminal_swd | tokenizer losses | 3 |
| 620 | 06-19 | L_flow | single_step_swd(8), edge(0.1) | OT cost | 3 |

---

## 详细演变

### 1. SA-Flow Era (01月): 纯FM-MSE

```
L = MSE(v_pred, x_style - x_content)

插值: x_t = (1-t)*x_content + t*x_style + noise*0.01
```

**特点**: 最简洁的Flow Matching，无任何辅助loss。OT匹配后成为标准CFM。

**问题**: 风格弱，因为没有显式style loss。FM-MSE只保证trajectory正确，不保证style质量。

### 2. LGT-X Era (01-28 ~ 02月): SWD为主

```
L = w_swd * L_swd + w_mse * L_mse + w_color * L_color + w_identity * L_identity

w_swd = 100, w_mse = 0.2
SWD patch sizes: [1,3,5,7,15], weights: [1,5,5,5,3]
```

**演变**:
- 01-20: 加频谱幅度Loss → 不稳定
- 01-24: 加MSE辅助保留内容 + 速度正则化 → 解决亮度变化
- 01-25: "balance loss" → 找到平衡点
- 02-10: **Cycle改MSE** → "风格确实好了，雾也解决了"

**关键发现**: 对抗loss(CycleGAN)在latent space不如MSE稳定

### 3. C-G-W Era (02月): SWD简化

```
L = w_swd * L_swd + w_color * L_color
- structure loss删除 (02-16, commit 54d120e: "structure loss完全没用")
- TV删除 (03-22, commit a4d6936: "TV可以扔了")
```

### 4. Cycle-NCE Era (03月): 多Loss膨胀

```
L = w_swd*SWD + w_color*color + w_identity*identity + w_nce*PatchNCE + ...

Color loss演变:
  pool_mse → 4 modes (pseudo_rgb_adain, pseudo_rgb_hist, latent_decoupled_adain, legacy_pool_mse)
  Winner: latent_decoupled_adain ("color 01效果极好")
  Channel weights: [2,1,1,1] (亮度通道权重2×)

SWD权重演变:
  w_swd: 30 → 150 → 120 → 250
  Patches: [3,5] → [1,3,5,9,15] → [7,11,15,19,25] → [1,3,5] → split micro/macro

Identity权重演变:
  w_identity: 2 → 30 → 35 → 0 (最终消失)
```

### 5. Schrödinger Bridge初始 (05-07): OMF Mode

```json
{
  "objective_mode": "omf",
  "w_kinetic": 0.0,
  "w_color": 15.0,
  "w_repulsive": 1.0,
  "terminal_swd_weight": 0.1,
  "bridge_sigma": 0.05
}
```

**OMF (One-Mode Flow)**: t=1固定，直接预测endpoint，多heuristic loss约束

### 6. SB Kitchen-Sink (05-08): Loss膨胀到9项

```
L = w_flow*L_flow + w_kinetic*L_kinetic + w_low_freq*L_low_freq_anchor
  + terminal_swd_weight*L_terminal_swd + w_color*L_color
  + w_nce*L_patch_nce + w_cycle*L_cycle + w_repulsive*L_repulsive

权重变化:
  terminal_swd: 0.1 → 25.0 (250× ↑)
  w_kinetic: 0.0 → 2.0
  w_color: 15.0 → 10.0
  w_repulsive: 1.0 → 0.1 (10× ↓)
```

**新增Loss**:
- `L_kinetic`: (velocity²).mean() → 防止速度爆炸
- `L_patch_nce`: PatchNCE对比loss → **摧毁风格** (0.674 vs 0.694)
- `L_cycle`: cosine lock cycle consistency → **可忽略** (0.693 vs 0.694)
- `L_low_freq`: 低频结构anchor → 只在OMF mode有效
- `L_semantic_swd`: SWD沿semantic key方向 → 保留

### 7. Black-Dot危机 (05-09): 数值爆炸

```
问题链: 全频SWD高权重 → velocity/endpoint极端 → NCE/rep放大 → NaN → 黑点

修复:
  _sanitize_tensor(): nan_to_num + clamp
  normalize_eps=1e-8, logit_clamp=50, velocity_clamp=20
  endpoint_clamp=24, similarity_clamp=50

Probe实验:
  +NCE:      clip_style=0.674 (↓0.020), LPIPS=0.434 → **摧毁风格**
  +cycle:    clip_style=0.693 (↓0.001), LPIPS=0.545 → 可忽略
  +repulsive: clip_style=0.695 (↑0.001), LPIPS=0.550 → 无帮助
```

### 8. Phase 1 Cleanup (05-19): 大清洗

**删除的** (全部已确认无收益或负收益):
| Loss | 行数 | 删除原因 |
|------|------|---------|
| L_color | ~32 | w=0.0, 导致black dots |
| L_repulsive | ~30 | negligible, 数值爆炸 |
| L_patch_nce | ~25 | 摧毁风格 |
| L_cycle | ~10 | negligible (0.001 diff) |
| L_low_freq | ~15 | 只OMF mode用, OMF已删 |
| L_low_freq_structure | ~15 | 从未被调用 |
| _compute_omf | ~253 | 整个OMF mode |
| _freq_split | ~8 | full-band SWD替代 |
| micro/macro SWD | ~160 | 简化 |

**保留的**:
| Loss | 权重 | 作用 |
|------|------|------|
| L_flow | 1.0 | 核心flow matching目标 |
| L_kinetic | 1.5 | 速度正则，防爆炸 |
| L_terminal_swd | 0.15 | 风格质量 |
| L_curvature | 0.0(可选) | 路径平滑 |

**简化后的loss公式**:
```
L = L_flow + 1.5*L_kinetic + 0.15*L_terminal_swd
```

### 9. 620 Spatial Bridge (06-19): 再简化

```
L = L_flow + single_step_swd_weight*L_swd + edge_weight*L_edge

single_step_swd_weight = 8 (后来试2)
edge_weight = 0.1
sigma = 0.02
```

**与SB的区别**:
- 无OT matching (OT在Euclidean space失败)
- 单步SWD (不是terminal_swd)
- 无kinetic energy (endpoint parameterization不同)
- 加edge loss保留结构

---

## SWD权重演变总结

```
01月: w_swd = 100, patches [1,3,5,7,15]
03月: w_swd = 30→150→120→250, patches [3,5]→[1,3,5,9,15]→[7,11,15,19,25]
03月: split w_swd_micro=0, w_swd_macro=80
05月: terminal_swd = 0.1→25→0.15
06月: single_step_swd = 8→2→0.15
```

## 关键教训

1. **对抗loss在latent space不稳定** — CycleGAN→MSE是正确方向
2. **NCE/repulsive/cycle等heuristic loss无收益或负收益** — 清洗掉是对的
3. **SWD是唯一有效的style loss** — 但权重需要仔细调
4. **Kinetic energy防止velocity爆炸** — 必须保留
5. **Loss越少越稳定** — 942→340行的清理后训练更稳定
6. **Color loss在latent space有专门设计需求** — latent_decoupled_adain比pool_mse好得多
7. **Identity loss最终被证明不必要** — w从2→30→35→0
8. **全频SWD>频率分解** — 更简单更快更稳定
