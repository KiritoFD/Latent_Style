# 训练策略与实验结果演变史

## 总览

| 阶段 | 时间 | Batch | LR | Epochs | 优化器 | AMP | 特殊策略 |
|------|------|-------|-----|--------|--------|-----|---------|
| SA-Flow | 01月 | 12→22 | 1e-4→2e-4 | 500 | AdamW | ✓ | OT匹配, 两阶段 |
| LGT-X | 01月 | 96→128 | 1e-4 | 600 | AdamW | ✓ | CFG(scale=4), label_drop=0.25→0.1 |
| C-G-W | 02月 | 96 | - | 60 | - | ✓ | 消融20组 |
| Cycle-NCE | 03月 | 256 | 1e-4 | 120 | AdamW | bf16 | GPU预加载, channels_last |
| SB | 05月 | 20-48 | 5e-5 | - | AdamW | ✓ | 两阶段(train→reflow) |
| Distinct5 | 06月 | 12-24 | - | 33 | - | ✓ | 11G VRAM cap, segmented training |
| 620 | 06月 | 4-80 | - | 1-10 | - | ✓ | 3060 12G, ablation 44+ configs |

---

## Batch Size演变

```
01月: 12 → 22 → 96 → 128 (LGT-X) → 240 (group conv + grad checkpoint)
02月: 96 (C-G-W)
03月: 256 (GPU预加载, 5-style)
04月: micro batch (效果大好) → 具体数字待确认
05月: 20-48 (SB, 受限于12G VRAM)
06月: 4-80 (620, base_dim=128时batch=24)
```

**关键发现**: "micro batch效果大好" (commit `58831eb6`, 04-02) — 小batch反而更好

## 学习率演变

```
01月: 1e-4 → 2e-4 (OT后加倍) → "增大速度会跑飞" → 减小
03月: 8.1e-4 → 5e-4 (C-G-W)
05月: 5e-5 (SB标准)
```

**关键教训**: "学习率太大跑飞了，下次loss上升先调整学习率" (commit `2883313b`)

---

## 实验结果时间线

### 01月: SA-Flow基线

| 实验 | clip_style | 备注 |
|------|-----------|------|
| SA-Flow v5 (2-style) | ~0.60 | 弱基线 |
| + OT matching | ~0.63 | 小幅提升 |
| LGT-X CrossAttn | ~0.68 | "风格强多了" |
| LGT-X 纯AdaGN | ~0.65 | 稳定但弱 |

### 02月: C-G-W消融

| 实验 | clip_style | 备注 |
|------|-----------|------|
| C-G-W base | ~0.667 | 低于前代 |
| Cycle→MSE | 改善 | "风格确实好了，雾也解决了" |
| overfit50 | 很好 | 过拟合信号 |
| Structure loss=0 | 无差异 | "完全没用" |

### 03月: LatentAdaCUT

| 实验 | clip_style | 备注 |
|------|-----------|------|
| decoder-D-160 | **0.720** | 前高水平 |
| CrossAttnAdaGN 64-token | 0.72 | 效果明显 |
| C-G-W backbone | 0.667 | 退步 |
| SWD hf | 负收益 | 高频SWD无用 |
| style_oa_5 | **0.72** | 好成绩 |

### 05月: Schrödinger Bridge

| 实验 | clip_style | LPIPS | 备注 |
|------|-----------|-------|------|
| SB base (omf) | 0.694 | 0.548 | 基线 |
| +NCE | 0.674 | 0.434 | **摧毁风格** |
| +cycle | 0.693 | 0.545 | 可忽略 |
| +repulsive | 0.695 | 0.550 | 无帮助 |
| Tokenizer base | 0.798/0.331 | - | 消融基线 |
| Low-cell b48 | 0.781/0.397 | - | **两面更差** |

### 06月: Distinct5 + 620

**Distinct5 LANCET**:

| 实验 | clip_style | LPIPS | 备注 |
|------|-----------|-------|------|
| solver_pc best (33ep) | 0.701 | - | **天花板** |
| SMoE translator (9ep) | 0.6728 | 0.3272 | 平衡 |
| I2SB topo_anchor σ=0.25 | **0.7197** | 0.7285 | 高style但内容崩溃 |
| I2SB latent_slerp (2ep) | 0.7120 | 0.4765 | SLERP路径 |

**620 Spatial Bridge (44+实验)**:

| 实验 | clip_style | LPIPS | WFI | 备注 |
|------|-----------|-------|-----|------|
| 620_swd12_b80 (8ep) | 0.6725 | 0.2968 | - | SWD宽度最优 |
| 620_film_formal (5ep) | 0.6723 | 0.2915 | 0.5037 | FiLM有效 |
| 620_film_v5_hd512 (1ep) | - | - | **0.3906** | WFI最优 |
| 620_lowswd_formal (2ep) | **0.6751** | 0.2781 | - | AP最高 |
| 620_lowmix05 (1ep) | 0.6765 | 0.3492 | - | transfer最高 |
| 620_intrinsic_v2 (8ep) | 0.6717 | 0.3678 | - | 内禀cross-attn |
| H7 SWD 8→2 | - | - | - | 缓解梯度冲突 |

**SaMST per-style基线**:

| 风格 | clip_style | LPIPS |
|------|-----------|-------|
| Baroque | 0.7234 | 0.2939 |
| Impressionism | 0.7361 | 0.2815 |
| Cubism | 0.7766 | 0.4270 |
| Symbolism | **0.7929** | 0.3339 |
| Art_Nouveau | 0.7694 | 0.3509 |
| **平均** | **0.7597** | **0.3374** |

---

## 关键指标演变

### clip_style (越高=风格越强)
```
01月: 0.60 (SA-Flow) → 0.68 (LGT-X cross-attn) → 0.65 (AdaGN)
03月: 0.72 (LatentAdaCUT 64-token) → 0.667 (C-G-W退步)
05月: 0.694 (SB base) → 0.798 (tokenizer, 消融基线)
06月: 0.701 (Distinct5 ceiling) → 0.67-0.68 (620)
```

### LPIPS (越低=内容保持越好)
```
05月: 0.548 (SB base) → 0.434 (+NCE, 但style被摧毁)
06月: 0.268 (620_swd20) → 0.368 (620_intrinsic)
```

### WFI (越低=白化越少)
```
06月: 0.50 (620_film_formal) → 0.39 (620_film_v5_hd512, 最优)
     → 但3 epoch后恶化: 0.43→0.45→0.47
     Seedream IDT ≈ 0.158 (参考)
```

---

## 白化定量等级

| 等级 | WFI范围 | 描述 |
|------|---------|------|
| 正常 | 0.50-0.65 | 正常风格化图像 |
| 轻微白化 | 0.68-0.72 | 轻微雾感 |
| 中等白化 | 0.73-0.78 | 明显雾感 |
| 严重白化 | >0.85 | 几乎全白 |

当前最优WFI=0.3906，在"正常"范围内，但训练后恶化到0.47。

---

## 训练硬件演变

| 时期 | 硬件 | 显存 | 限制 |
|------|------|------|------|
| 01月 | 本地GPU | - | 初期开发 |
| 05月 | 远程3060 WSL | 12GB | batch≤48 |
| 06月 | 远程3060 WSL | 12GB | 11GB VRAM cap, batch≤24(dim=128) |
| 06月 | 本地+远程 | - | 本地eval, 远程train |

---

## 关键教训

1. **大batch不一定好** — micro batch反而效果大好
2. **clip_style不是可靠的唯一指标** — 白化图可能clip_style很高
3. **WFI是白化的直接衡量** — 必须与clip_style联合看
4. **训练时间≠效果提升** — 3 epoch WFI恶化，更多训练加剧白化
5. **0.70是隐含天花板** — LANCET/Distinct5卡在0.70，620也在0.67-0.71
6. **Per-style模型(0.76) >> 通用模型(0.67-0.71)** — 差距0.05-0.09
7. **I2SB高σ=0.25可达0.72但LPIPS=0.73** — 风格强度和内容保持不可兼得
8. **FiLM endpoint是目前唯一通过WFI<0.40门的方案** — 但不稳定
