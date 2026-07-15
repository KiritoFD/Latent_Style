# 710 Phase B-C 实验结论汇总（2026-07-10 完成）

## 核心结论

### 1. 有效组件验证（最小消融）

| 组件 | 消融结论 | 保留/删除 |
|------|----------|-----------|
| Rectified Flow Matching | 核心机制，有效 | ✓ 保留 |
| Haar Wavelet Decomposition | w/o: CLIP-S -0.016，有效 | ✓ 保留 |
| DWT Routing | p=0.8 → DINO-C +0.0505, LPIPS -0.0755，非常有效 | ✓ 保留 |
| Routing Probability | p=0.8 (stochastic) 优于 p=1.0 (deterministic)，正则化作用 | ✓ p=0.8 |
| LL Supervision Weight | w_ll=0.3 is optimal；w_ll=1.0 → LPIPS +0.0181 (worse)；w_ll=0 → DINO-C +0.0239 (better) but DINO-S -0.0037 (worse) | ✓ 0.3 |
| Cross-Attention | cross-attn off → DINO-S -0.005，虽然 trade-off DINO-C +0.038，但风格下降可测；保留 | ✓ 保留 |
| Bridge Noise σ=0.02 | σ=0 → DINO-S +0.0028 but DINO-C -0.005，内容略退化；保留默认 | ✓ 0.02 |
| Endpoint WCT | 移除 → DINO-S -0.0057，有效 | ✓ 保留 |

### 2. DWT Routing 结论

**DWT Routing** 是 WEAVE 最有效的架构创新：
- 将高频子带（LH/HL）路由给 cross-attention 风格注入
- LL 子带绕过 cross-attention，结构性锁死内容
- 实测：DINO-C 提升 +0.0505，LPIPS 降低 -0.0755 → 显著双赢

**Routing 概率 p=0.8 (随机路由)** 优于 p=1.0 (确定性路由)：
- p=0.8：CLIP-S 0.7216, DINO-S 0.4277
- p=1.0：CLIP-S 0.7163, DINO-S 0.4267
- 随机 routing 提供正则化，避免过拟合

### 3. AdaLN 消融审计（2026-07-12）

**结论：global/local HF AdaLN 均无实质收益**

| Variant | CLIP-S | LPIPS | DINO-S | DINO-C | ΔDINO-S vs control |
|---------|--------|-------|--------|--------|-------------------|
| b144 control | 0.7219 | 0.3208 | 0.4786 | 0.7813 | — |
| global HF AdaLN | 0.7221 | 0.3225 | 0.4793 | 0.7818 | +0.0007 |
| local HF AdaLN | 0.7213 | 0.3202 | 0.4801 | 0.7811 | +0.0015 |
| local 2-layer AdaLN | 0.7213 | 0.3221 | 0.4796 | 0.7810 | +0.0010 |

- 最大增益仅 +0.0015，低于 baseline 方差（±0.004）
- 扩展到两层后增益回落，LPIPS 变差 → 容量增加无收益
- 所有 AdaLN 变体均已**删除**，结论：**Decoder 中任何位置的 AdaLN 均无效**

### 4. ASG 突破（Adaptive Style Gate）

**关键发现：训练侧空间自适应 gate 打破推理天花板**

| 配置 | epochs | CLIP-S | LPIPS | DINO-S | DINO-C | ΔDINO-S |
|------|-------|--------|-------|--------|--------|---------|
| S0 WEAVE (10ep) | 10 | 0.7298 | 0.4631 | 0.4421 | 0.6951 | baseline |
| wct_ll05 (推理天花板) | — | 0.7330 | 0.5422 | 0.4614 | 0.5937 | +0.0193 |
| **T1 ASG (训练侧)** | 5 | **0.7261** | **0.3354** | **0.4843** | **0.7692** | **+0.0422** |

**核心突破**：
1. 打破推理天花板：DINO-S 从 0.4617 → 0.4843 (+0.0226)
2. 内容保持大幅改善：DINO-C 从 0.6951 → 0.7692 (+0.0741)，LPIPS 从 0.4631 → 0.3354 (-0.1277)
3. CLIP-S 几乎不变：-0.0037，风格质量未牺牲
4. 训练效率极高：23s/epoch，5 epoch ≈ 2 分钟，VRAM ≈ 2.91GB

**机制解释**：
- 推理时标量 gate 已进入局部最优，任何方向扰动都退化
- ASG 将标量 gate 升级为 content-dependent 空间 gate map：平坦区域少受 style 干扰，纹理区域增强风格转移
- MLP 零初始化，训练初期等价于标量 gate，随训练逐渐学习空间自适应调制

### 5. 当前训练 loss 构成（已清理后）

**有效且可解释的训练旋钮：**

| Knob | Current Value | Expected Effect |
|------|---------------|-----------------|
| `spectral_w_ll` | 0.3 | 控制低频迁移/内容锚；`0` 损害内容平衡，`1` 更易牺牲 LPIPS |
| `spectral_w_lh` | 1.0 | 中频速度监督权重 |
| `spectral_w_hl` | 1.0 | 高频速度监督权重 |
| `bridge_sigma` | 0.02 | 桥中间态噪声，影响泛化 |
| `loss_type` | `mse` | 可切换 Huber，但未验证 |
| `structure_aligned_target` | `false` | 功能保留，未进入 baseline |
| `subband_time_schedule_enabled` | `false` | 功能保留，需单独 A/B |

**已退休（不再进入反传）：**
- SWD（任何形式：single-step/terminal/contrastive）
- Edge loss / Low-pass anchor loss
- Endpoint content/style auxiliary losses
- Gram/moment/variance 类旧 loss

当前 `src/flow.py` 只计算 **spectral flow matching MSE 三带损失**：
$$
\mathcal{L} = 0.3 \mathcal{L}_{LL} + 1.0 \mathcal{L}_{LH} + 1.0 \mathcal{L}_{HL}
$$

### 6. DINO 评估协议（统一 canonical）

**正确定义（自 710 审计后）：**

- **DINO-S** = $\max_{ref} \cos(\text{CLS}(gen), \text{CLS}(target\_style\_ref))$ — 风格相似度，↑ 越好
- **DINO-C** = $\cos(\text{CLS}(gen), \text{CLS}(source))$ — 内容保持，↑ 越好
- **DINO-structure** = $\text{MSE}(\text{SSM\_patch}(gen), \text{SSM\_patch}(source))$ — 结构失真，↓ 越好

Backbone: DINOv2-small; Preprocessing: 224 bicubic resize + center crop + ImageNet normalization.

**错误定义（已废弃）：**
- 旧脚本误将 `1 - patch self-similarity MSE` 当作 DINO-C，量纲不对，现已撤回

### 7. 判定规则（四指标 Pareto）

1. 首先要求 `CLIP-S` 高于 IDT floor
2. 若 `CLIP-S`/`DINO-S` 改善，则要求 `LPIPS`/`DINO-C` 退化不超过基线种子标准差
3. 若 `LPIPS`/`DINO-C` 改善，则要求 `CLIP-S`/`DINO-S` 退化不超过基线种子标准差
4. 只在至少 3 个 seeds 的均值和标准差上做结论
5. 停止标准：不满足 → 删除代码

### 8. Infra 优化结论（8GB 环境）

**训练：**
- 删除每 step `.item()` → 36.3s → 29.7s/epoch (+18% speedup)
- 推荐 batch 24, bf16, fused AdamW → 5 epochs ≈ 153s on 8GB

**推理（750 images, 8-step）：**
- 聚合 latent 后批量 VAE decode → 53.75s → 34.97s VAE decode (-35%)
- 总端到端：93.88s（桥接 57.42s + VAE 34.31s）
- 推荐：固定 TorchScript VAE decoder, decode batch 16, async PNG save

**瓶颈：** 当前主瓶颈是 VAE decode，不是桥接。

### 9. 当前 Minimal 代码基线

**Active Source Files:**

| File | Responsibility |
|------|----------------|
| `src/model.py` | WEAVE network, velocity heads, ODE solver, endpoint alignment |
| `src/flow.py` | Training interpolation and band-weighted flow-matching loss |
| `src/blocks.py` | Residual and cross-attention block |
| `src/wavelet.py` | Haar DWT/IDWT and subband schedules |
| `src/style.py` | DINO patch/style-memory projection |
| `src/trainer.py` | Training loop, checkpointing, logging |
| `src/config_schema.py` | Typed experiment configuration |
| `src/run.py` | CLI entry point |
| `src/style_families.py` | Checkpoint/config validation helpers |

**Baseline Configuration (`710_b0_weave_d5.json`):**

- 3 learned velocity heads: LL, LH, HL (HH closed)
- Band weights: `0.3 / 1.0 / 1.0`
- Plain band-wise flow matching only (no auxiliary losses)
- DWT routing probability `p=0.8`
- Bridge noise `sigma=0.02`
- `spatial_fiber` endpoint alignment, `endpoint_adain_scale=1.0`
- Endpoint alignment per step (not only last step)

---

## 下一步（已完成于 712/713）

1. ✓ ASG 已确认突破，进入 main table
2. ✓ DINO-S 天花板已验证在 0.48 ± 0.003（不引入 DINO 训练 loss）
3. ✓ 推理时 Endpoint AdaIN 缩放可推高到 0.4859 (α=2.0)，CLIP-S 略有下降
4. ✓ 代码清理完成：所有失败实验（10 轮共 30+ 方向）已删除

---

## 原始实验数据

- 所有原始结果：`I:/Github/Latent_Style/SchrodingerBridge/exp/710_*`
-  canonical DINO 结果：`I:/Github/Latent_Style/SchrodingerBridge/exp/710_canonical_dino_results.txt`
-  进度日志：`docs/dino_s_break/state/progress.json`
