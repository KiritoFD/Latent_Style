# Round 1 诊断报告 — 620 Spatial Bridge 实验结果分析

> 日期: 2026-06-19. 基于远程 RTX 3060 WSL 上的全部 7 个变体 × 8 epoch 训练结果.

---

## 1. 实验结果汇总

| Variant | Best clip_style (transfer) | E8 LPIPS | E8 delta_idt | E8 all_pairs_style | Status |
|---------|---------------------------|----------|--------------|-------------------|--------|
| base_swd8 | **0.6720** | 0.2900 | 0.0321 | 0.7055 | ✓ LPIPS极好 |
| swd4 | 0.6706 | **0.2794** | 0.0306 | 0.7050 | ✓ 最低LPIPS |
| swd12 | **0.6725** | 0.2968 | 0.0325 | **0.7056** | ✓ 最高style |
| adapter | 0.6715 | 0.2916 | 0.0315 | 0.7046 | ✗ 无提升 |
| moe | 0.6711 | 0.2906 | 0.0312 | 0.7048 | ✗ 无提升 |
| gate12 | 0.6714 | 0.2918 | 0.0315 | 0.7045 | ✗ 无提升 |
| lowmix05 | 0.6765 | 0.3492 | 0.0366 | 0.7059 | ⚠ LPIPS崩 |

**关键观察**:
- 所有变体的 clip_style 都在 0.668–0.677 区间, **远低于目标 0.72+**
- LPIPS 非常好 (0.28–0.30), 说明垂直 FM 在生效
- adapter/moe/gate12 **没有比 base_swd8 更好的 style 分数** — 增加容量无效
- delta_idt 约 0.030–0.032 — 与老 LBM baseline (0.030) 几乎一致
- lowmix05 (low_anchor=0.5) 稍微提升 style (0.677) 但 LPIPS 崩到 0.349 — 水平泄漏

---

## 2. 根因诊断

### 2.1 网络容量严重不足 (最关键)

当前模型只有 **1.55M 参数**, 其中:
- `style_conditioner`: 1.34M (86%) — DINO 投影层
- `blocks` (4 层, dim=64): **0.18M (12%)**
- `time_proj`: 0.02M
- `input_proj` + `out`: 0.005M

**问题**: 4 层 dim=64 的 block 只有 **183K 参数** 来执行"从 256 个 DINO patch tokens 中学习风格纹理搬运". 对比:
- StyleShot 的 style adapter: ~10M
- SaMAM 的 UNet: ~50M+
- 仓库 legacy LANCET bridge (dim=128, 多层): 数 M

dim=64 在 latent 空间 (4×64×64) 是极其小的瓶颈. 64 维通道意味着每个空间位置的表示容量只有 64 floats — 而 DINO patch tokens 有 384 维. 即使投影到 64 维, 信息也被强制压缩了 6 倍.

**为什么 adapter/moe 没帮助**: 它们增加的是 K/V 投影的容量, 但 Q 仍然只有 64 维. Cross-attention 的输出维度受限于 `head_dim × num_heads = dim = 64`. 无论 K/V 有多大, 最终 `attended` 向量还是 64 维, 瓶颈在 Q 侧.

### 2.2 Self-Attention 被完全移除

当前 block **没有 self-attention** — 只有 cross-attention (content Q × style K/V). 这意味着:
- 空间位置之间无法通信 (无法传播笔触模式)
- Cross-attention 的输出是独立的空间位置 × style tokens 的加权和 — 没有空间一致性
- 笔触纹理需要相邻位置的协调 (如"一条笔触跨多个 patch"), 没有自注意力就无法学习

### 2.3 SWD 的有效梯度信号可能被模型容量瓶颈吞噬

SWD loss 确实在提供 style 梯度 (它比 FM loss 大 8-12 倍), 但如果模型只有 64 维通道和 4 层, 这些梯度可能只是在微调投影权重, 而不是在学真正的纹理搬运.

### 2.4 数据集偏小

5000 张训练图 (5 styles × 1000) 对于从零学习一个新风格的 cross-attention 机制来说偏小. 尤其考虑到每种 style 只有 1000 张图, 每张图的 DINO 有 256 个 patch tokens — 有效训练样本只有 1000 × 256 = 256K 个 patch-style pair per style.

### 2.5 `full_eval_defer_until_training_end: true` 的时机问题

所有训练 epoch 的 eval 都被延迟到最后才跑 (通过 I2SB 8 步推理). 这意味着:
- 训练期间看不到任何 eval 反馈
- 无法在训练中途判断是否需要早停或调参
- 但这本身不是 style 不高的原因, 只是让实验迭代变慢

---

## 3. 修正方案 (优先级排序)

### P0: 扩大网络容量 (必须)

**dim=64 → dim=128**, 保持其他不变. 这让 block 参数量从 183K 增加到 ~730K (4×). 同时:
- 增加 `num_res_blocks` 从 4 到 6
- 保持 `style_attn_num_heads=4`, head_dim 从 16 增到 32

预估总参数: ~3-4M. 在 RTX 3060 (12GB) 上完全可行 (batch=80 × 4ch × 64×64 latent = ~21MB 输入).

### P1: 恢复 Self-Attention

在每个 block 中加入 self-attention (content Q × content K/V):
```
h → norm → AdaLN(time) → Self-Attn → residual
    → Cross-Attn(style) → residual
    → FFN → residual
```
这是标准 DiT 架构, 数学上保证空间一致性.

### P2: 训练时每个 epoch 做 eval (不是延迟到最后)

设置 `full_eval_defer_until_training_end: false`. 虽然每个 epoch eval 会多花 ~90s, 但对实验迭代效率至关重要.

### P3: 考虑增加训练 epoch

当前所有变体在 8 epoch 后 style 曲线仍在缓慢上升 (未收敛). 增加 epoch 到 16-24 可能让 style 再涨 0.01-0.02.

### P4 (可选): 学习率调整

当前 lr=2e-4 对 dim=64 的小模型可能偏大 (容易震荡). dim=128 可以用 lr=1e-4 或配合 cosine schedule.

---

## 4. 具体配置改动

```jsonc
{
  "model": {
    "base_dim": 128,           // 64 → 128
    "num_res_blocks": 6,       // 4 → 6
    "style_attn_num_heads": 8  // 4 → 8 (head_dim=16, 保持不变)
  },
  "training": {
    "learning_rate": 0.0001,   // 2e-4 → 1e-4
    "num_epochs": 16,          // 8 → 16
    "full_eval_defer_until_training_end": false  // 每个 epoch eval
  }
}
```

### 4.1 Block 架构改动 (blocks620.py)

在 Cross-Attention 之前加入 Self-Attention:
```python
def forward(self, x, *, time_emb, style_tokens):
    # 1. AdaLN(time)
    # 2. Self-Attention (content × content)
    # 3. Cross-Attention (content × style)  
    # 4. FFN
```

### 4.2 Batch Size 预估

dim=128, 6 层, batch=80:
- 模型: ~4M params × 4 bytes = ~16MB
- 激活: 80 × 128 × 64 × 64 × 6 层 ≈ 1.5GB
- Cross-attention: 80 × 4096 × 256 × 128 维 ≈ 1GB
- 总计: ~3-4GB (训练时含梯度/优化器 ~6-8GB)
- RTX 3060 12GB → batch=80 可行, 但建议 batch=64 保安全

---

## 5. 不改的东西

- **Vertical FM**: 保留. LPIPS 0.29 的结果表明它工作良好.
- **SWD 单步**: 保留, 但扫描范围改为 {4, 8, 12} 而不是只跑固定值.
- **I2SB 推理**: 保留 NFE=8 sigma=0.02.
- **离线 DINO 配对**: 保留, 基础设施已就绪.
- **DINO frozen**: 第一轮保留 frozen. 如果容量提升后仍卡, 再考虑 adapter.
- **dataset.md 的 stratified pairing**: 当前 Distinct5 数据集是平衡的 (每 style 1000), 暂不需要.
