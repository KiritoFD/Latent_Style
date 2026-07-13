# AdaLN 与 Loss 审计（2026-07-12）

## 结论

- 不用 AdaLN 替换当前 endpoint AdaIN。endpoint 的 spatial-fiber AdaIN 是已验证的直接统计对齐；AdaLN 只能作为训练期的补充调制。
- 本轮 global / local high-pass AdaLN 都没有得到 strict four-metric win。局部一层只有很小的 DINO-S 波动，扩展到两层后退化，代码已删除。
- 当前训练的真实目标是三带 spectral flow matching，不是 SWD、Gram、edge、content 或 endpoint auxiliary loss。旧配置中的这些 loss 字段会被 schema 丢弃，不能再作为可调旋钮。

## 受控 AdaLN 屏幕

共同条件：Distinct5、seed 42、batch 144、5 epochs、170 optimizer updates、相同 750 张 canonical 输出、CLIP/LPIPS all-pairs 与同图集 DINO。所有分支保持 endpoint `spatial_fiber` AdaIN、`w_LL=0.3`、`sigma=0.02` 不变。

| Variant | Params | CLIP-S | LPIPS ↓ | DINO-S | DINO-C |
|---|---:|---:|---:|---:|---:|
| b144 control | 903,248 | 0.721909 | 0.320849 | 0.478624 | 0.781324 |
| global HF AdaLN, late-1, rank 64 | 932,497 | 0.722116 | 0.322485 | 0.479270 | 0.781770 |
| local HF AdaLN, late-1, rank 64 | 961,617 | 0.721320 | 0.320159 | 0.480076 | 0.781052 |
| local HF AdaLN, late-2, rank 64 | 1,019,986 | 0.721340 | 0.322081 | 0.479646 | 0.781030 |

相对 control，local late-1 的 DINO-S 为 `+0.001452`，但 CLIP-S 为 `-0.000589`；扩展到 two-layer 后 DINO-S 回落 `-0.000430`、LPIPS 变差 `+0.001922`（相对 local late-1）。全局版本的最大 DINO-S 增益也只有 `+0.000646`。

局部模块不是零梯度：最终 gain 从 `0.100` 学到 `0.115`，global/local projection 的输出层均离开零初始化。故负结论来自收益不足，而非模块未接入或未训练。

该屏幕用于同 batch 的结构比较；它不能和 b24/5epoch 基线直接比较收敛质量，因为 b144 只有约六分之一的优化步数。

清理后，未获晋级的 AdaLN 与无使用者的 global style projection 均已移除；当前 batch-144 最小模型为 `873,680` 参数，并可显式兼容加载上述旧 checkpoint。

## AdaIN 与 AdaLN 的职责

- **Endpoint AdaIN**：在 ODE 结束时把生成 latent 的 spatial fiber mean/std 对齐到目标风格，直接改变最终输出统计；历史消融表明移除它明显损失风格。
- **High-pass AdaLN（已删除）**：在残差块内对 LH/HL/HH 做有界的 channel-wise scale/shift，global 版本由 pooled style token 生成，local 版本由已有 cross-attention 的空间输出生成；它不改 LL，因此理论上更安全，但本轮没有足够收益。
- **局部性**：local AdaLN 的确可以复用现有 attention map，避免另建 dense style branch；但这一实现的 capacity 扩展从 1 到 2 个 late blocks 没有帮助。

因此下一次若重新考虑 learned modulation，必须先有三 seed 的 single-layer local 复验，并以 `DINO-S` 提升超过 baseline 方差、且不劣化其余三项为门槛；否则不重新引入该代码。

## 当前真实 Loss 与旋钮

`src/flow.py` 当前只计算：

`L = 0.3 L_LL + 1.0 L_LH + 1.0 L_HL`，其中每项是对应 Haar velocity 的 MSE flow-matching loss。当前 baseline 的 HH velocity head 关闭，因此 `spectral_w_hh=2.0` 不参与损失或推理；只在未来 HH-head false/true matched A/B 中才有意义。

有效且可解释的训练旋钮：

| Knob | 当前值 | 预期影响 |
|---|---:|---|
| `spectral_w_ll` | 0.3 | 控制低频迁移/内容锚；历史 `0` 损害内容与风格平衡，`1` 更易牺牲 LPIPS。 |
| `spectral_w_lh`, `spectral_w_hl` | 1.0, 1.0 | 控制两类中高频速度监督，是当前主要风格学习信号。 |
| `bridge_sigma` | 0.02 | 对桥中间态注入小噪声，影响泛化与轨迹平滑。 |
| `loss_type` | `mse` | 可切换为 Huber，但尚未做同协议验证。 |
| `structure_aligned_target` | false | 功能仍在，但未进入当前 baseline；它只改变训练 target，不是额外 loss。 |
| subband time schedule | false | 功能仍在，按时间重加权 spectral FM；需单独 A/B。 |

已退休、不会进入反传的字段包括 SWD、terminal/single-step SWD、endpoint content/style weight、style-contrastive SWD、edge/content anchor、Gram/moment/variance 类旧 loss。它们现在由 `src/config_schema.py` 的 retired-key 过滤，训练日志也不再伪报 `w_content/w_style`。

## 下一步顺序

1. **先校正大 batch 收敛**：b144 每 epoch 覆盖相同样本量但 optimizer steps 大幅减少；先在不改结构的 b144 control 上测试 `lr=5e-4`，再决定是否把大 batch 作为主训练设置。
2. **推理侧 endpoint A/B**：使用现有代码严格比较 `spatial_fiber` 对 `per_subband` AdaIN；仅当 DINO-S 严格晋级时，再扫描 LH/HL/HH 分频强度。
3. **训练侧最小 A/B**：固定 endpoint 后，做 HH head `off/on`；只有 HH head 开启时才扫描 `spectral_w_hh`。
4. **LL 最后再动**：如需释放风格空间，只比较 `w_LL=0.1/0.3`，不重新引入 SWD 或无效的 endpoint loss。
5. **三 seed 门槛**：任何超过 100 行的新模块都必须在三 seed 四指标 Pareto 前沿中胜出，否则删除。
