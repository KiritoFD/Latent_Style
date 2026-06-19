# 620 下一步实验计划

> 当前最优: swd16, vl=0.04, e5 → clip_style=0.7051, LPIPS=0.2935
> 目标: style ≥ 0.72, LPIPS ≤ 0.30

---

## 为什么 SWD16 突破了 0.70

620 架构解决了 619 诊断的 3 个致命缺陷:

| 619 缺陷 | 620 如何解决 | 效果 |
|---------|-------------|------|
| OT 在线不稳定 | DINO top-k 离线预配对 (跨 epoch 目标稳定) | 消除均值坍缩的配对跳变源 |
| 伪 Cross-Attention | True Cross-Attention: DINOv2 256×384 空间特征序列 → K,V | 风格信息量从 KB→400KB |
| 训练 ODE 展开 | 单步 SWD: `SWD(ẑ₁, z_s)` 替代 `integrate()` 展开 | 消除梯度爆炸, 释放 SWD 梯度 |

**SWD=16 是关键**: 单步 SWD 不受 ODE 展开的 torch.nan_to_num 截断 → 强风格梯度可传播 → style 从 0.67 跳到 0.705.

---

## Phase 4: 逼近目标 (预计 8-10 实验, ~4h)

### P4.1: SWD 权重精调 (最高优先级)

SWD=16 已经接近上限，精调 SWD∈{14, 15, 16, 17, 18}，找最优值。每个 sweep 2 个 epoch (vlen=0.04), ~15min/实验.

```bash
for swd in 14 15 16 17 18; do
    python launch --variant swd${swd} --vlen 0.04 --epochs 5
done
```

**预期**: SWD=16 附近 ±0.5 为最优. 过高→LPIPS升, 过低→style降.

### P4.2: Style Encoder 的投影维度

DINOv2 输出 384-dim，投影到 UNet 的 Cross-Attention 维度 `d_model`. 当前默认 256. 增大→更多风格信息.

```bash
for d_model in 256 384 512; do
    python launch --variant swd16 --d-model $d_model --vlen 0.04 --epochs 5
done
```

**预期**: d_model=384 或 512 可能带来 0.005-0.01 style 提升.

### P4.3: Cross-Attention 注入层数 (当前可能是单层)

当前可能只在 bottleneck 做 CrossAttn. 改为多层注入→多尺度风格.

```bash
# 单层 (bottleneck only)
python launch --variant swd16 --cross-attn-layers "bottleneck"

# 多层 (所有 decoder blocks)
python launch --variant swd16 --cross-attn-layers "all"

# decoder only
python launch --variant swd16 --cross-attn-layers "decoder"
```

**预期**: "all" (所有层) 应提供最佳多尺度风格注入.

---

## Phase 5: 消融与稳定化 (如果 P4 达到 0.72)

### P5.1: 收敛曲线延长

当前只到 e5. 如果 e5 是峰值, 延长到 e20 确认不会退化.

```bash
python launch --variant swd16 --vlen 0.04 --epochs 20
```

### P5.2: 消融 DINO 预配对

对比: DINO top-k 预配对 vs 随机配对 (Independent Coupling).

```bash
# 随机配对对照
python launch --variant swd16 --pairing random --vlen 0.04 --epochs 5
```

**预期**: 随机配对 style 低 0.02-0.03, 但 LPIPS 可能更好→量化 DINO 预配对的贡献.

### P5.3: 消融 Cross-Attention

对比: True Cross-Attention (DINO 空间特征) vs AdaLN-only (全局 1D).

```bash
python launch --variant swd16 --style-injection "adain_only" --vlen 0.04 --epochs 5
```

---

## 如果 P4+P5 仍未达 0.72

### 备选路线 A: 增大学习率

当前 lr 可能偏保守. e5 已经 peak 说明模型收敛快但饱和也快. 增大 lr→更快到达更高点.

```bash
for lr in 2e-4 5e-4; do
    python launch --variant swd16 --lr $lr --vlen 0.04 --epochs 10
done
```

### 备选路线 B: Style Encoder 从冻结→微调

DINOv2 当前冻结. 最后几层可微调→针对 WikiArt 风格特化.

```bash
python launch --variant swd16 --dino-trainable "last_2_blocks" --vlen 0.04 --epochs 10
```

### 备选路线 C: 多参考图聚合

每 content 用 3 张 DINO top-k 风格图→Cross-Attention 看到更多风格变体.

---

## 时间预算

| Phase | 实验数 | 时间 |
|-------|:---:|------|
| P4.1 SWD 精调 | 5 | ~1.5h |
| P4.2 d_model | 3 | ~1h |
| P4.3 CrossAttn layers | 3 | ~1h |
| P5 消融 | 3 | ~1h |
| 备选 | 2-4 | ~1-2h |
| **总计** | **~16** | **~5-6h** |
