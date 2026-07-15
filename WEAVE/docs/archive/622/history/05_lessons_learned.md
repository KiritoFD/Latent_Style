# 教训总结：什么有效，什么无效

## 按收益分类

### ✅ 明确有收益的改动

| 改动 | 阶段 | 量化证据 | 原理 |
|------|------|---------|------|
| OT匹配(Hungarian) | 01月 | clip_style 0.60→0.63 | 减少trajectory crossing |
| Cross-attention (多token) | 03月 | clip_style 0.667→0.72 | 64-token vocabulary有选择性 |
| Sharpen scale=2.5 | 03月 | 防止soft attention | softmax logits放大→peaky attention |
| Cycle→MSE | 02月 | "风格确实好了，雾也解决了" | 对抗loss在latent space不稳定 |
| 去掉InstanceNorm | 04月 | aent 0.99→正常 | IN白化→均匀attention→无风格 |
| Zero-init gate | 03月 | 训练稳定 | 从AdaGN-only渐进学习attention |
| Micro batch | 04月 | "效果大好" | 大batch过度平均化 |
| DINOv2 open-set style | 06月 | 打破固定风格数限制 | 通用style encoder |
| FiLM endpoint head | 06月 | WFI 0.50→0.39 | style信号直接调制endpoint |
| SWD loss | 全程 | 唯一有效的style loss | 统计距离直接优化分布 |
| Kinetic energy loss | 05月 | 防止velocity爆炸 | 速度正则 |
| Phase 1 cleanup | 05月 | 训练更稳定 | 删除9项heuristic→3项核心loss |
| Q/K twin-norm | 04月 | 修复attention | 同一normalization才能匹配 |
| GPU预加载 | 03月 | batch 12→256 | 大batch前提 |
| Color loss (latent_decoupled_adain) | 03月 | "效果极好" | 通道级统计匹配 |

### ❌ 明确无收益或负收益的改动

| 改动 | 阶段 | 量化证据 | 原因 |
|------|------|---------|------|
| DiT (PatchEmbed) | 01月 | 训练失败 | 6D reshape bug, 过早尝试 |
| 1-token Cross-attention | 01月 | softmax恒=1.0 | 1 token无选择性 |
| 共享StyleController | 01月 | Layer Collapse | 所有层相似参数 |
| 全频SWD高权重 | 05月 | black-dot | 推velocity到极端→NaN |
| PatchNCE | 05月 | clip_style 0.694→0.674 | 摧毁风格信号 |
| Repulsive loss | 05月 | negligible (0.001 diff) | 无帮助 |
| Cycle consistency (cosine lock) | 05月 | negligible (0.001 diff) | 无帮助 |
| Structure loss | 02月 | "完全没用" (3次独立验证) | identity loss已隐含结构约束 |
| TV loss | 03月 | 可删除 | 不影响结果 |
| OMF mode | 05月 | 不稳定 | 被Flow Matching替代 |
| Frequency split (micro/macro) | 05月 | 简化后更好 | 增加复杂度无收益 |
| C-G-W backbone | 03月 | clip_style 0.72→0.667 | 退步 |
| IN on attention Q/K | 04月 | aent=0.99 | 白化features=均匀attention |
| OT in Euclidean space | 06月 | PureLatentSpatial=ZERO ROI | 需要结构指纹 |
| 高频SWD | 03月 | 负收益 | 风格在低频 |
| 大学习率 | 01月 | "跑飞了" | 不稳定 |
| 3+ epoch训练 | 06月 | WFI 0.43→0.47 | 训练越久白化越严重 |
| Low-cell probe | 05月 | 两面更差 | 弱cell重采样无收益 |
| Diff-Gram (可微Gram+黎曼) | 02月 | style_swd=**0.0** | 完全失败 |
| Gram-Moment+Semigroup | 02月 | +5507MB, 无改善 | 计算不可行+效果差 |
| Classifier-guided training | 02月 | clip_style最高0.59 | 分类器≠style质量 |
| Proxy CNN filtering | 01月 | IoU=0.247 | 完全失败 |
| Text conditioning (620) | 06月 | T5 vs no-T5差**0.001** | gate=0.048时无效 |
| 模型容量增加(64→128) | 06月 | clip_style差**0.001** | 容量不是瓶颈 |
| StyleFiLM on/off | 06月 | 无差异 | 1-epoch无信号 |

### ⚠️ 有信号但未充分验证

| 改动 | 阶段 | 信号 | 未验证原因 |
|------|------|------|-----------|
| Endpoint lowhigh分离 | 06月 | WFI改善 | 3 epoch后恶化 |
| Gate init=0.3→0.5 (vs 0.05) | 06月 | gate值仍收敛到0.048 | **gate_init不改变最终值** |
| Pre-CrossAttn FiLM | 06月 | line 237-243 | 代码存在但未被config激活 |
| I2SB σ=0.25 | 06月 | clip_style=0.72 | LPIPS=0.73内容崩溃 |
| SLERP路径 | 06月 | clip_style=0.71 | 仅2 epoch |
| SWD weight 8→2 | 06月(H7) | 缓解梯度冲突 | 未完成完整训练 |
| Fiber-SDE σ=0.08 (不训练) | 06月 | **clip_style=0.711** | 纯ODE, 比训练后更好 |
| AnisoStokes ClampRelease | 06月 | clip_style=0.701, LPIPS=0.475 | 最佳XPred平衡 |
| SMoE translator | 06月 | clip_style=0.67 | 仅9 epoch |
| Domain style (vs Instance) | 02月 | **5.77×更有效** | 仅在Style8_Moment+SWD验证 |

---

## 按问题分类的核心教训

### 问题1: 风格注入太弱

**历史**:
- 1-token cross-attn → softmax=1.0 (01月)
- 共享StyleController → Layer Collapse (01月)
- C-G-W去掉StyleAdaptiveSkip → clip_style退步 (03月)
- Distinct5 ceiling=0.701 (06月)
- 620 style_gate=0.048, cross-attn entropy=6.24 (06月)

**根因**: style信号在注入点被衰减
**已验证的解法**: 64-token vocabulary + sharpen + zero-init gate
**待验证**: Pre-CrossAttn FiLM, 更大gate_init, DINO longer training

### 问题2: 白化 (Endpoint Shrinkage)

**历史**:
- 04月: InstanceNorm白化 → 均匀attention
- 06月: 620 endpoint只走16%目标方向 (latent_alpha≈0.163)
- 06月: 高频方向为负 (high_alpha≈-0.050)
- 06月: 3 epoch WFI恶化

**根因**: endpoint参数化时，模型学到了shrinkage策略（往source靠拢更安全）
**已验证的解法**: FiLM endpoint (WFI 0.39, 但不稳定)
**待验证**: 更强style条件, 直接predict style residual

### 问题3: 指标不可靠

**历史**:
- clip_style高但图白化 (06月)
- LPIPS低但style被摧毁 (+NCE, 05月)
- WFI是白化的直接衡量，但与clip_style不总一致

**教训**: 至少需要clip_style + LPIPS + WFI三个指标联合判断

### 问题4: 训练不收敛/不稳定

**历史**:
- DiT bug (01月)
- MSE爆炸 (01月)
- Black-dot (05月)
- WFI恶化 (06月)

**已验证的解法**:
- 从简单架构开始验证pipeline
- Loss越少越稳定
- Kinetic energy防止爆炸
- Clamp/sanitize作为安全网
- Zero-init保证起步稳定

---

## 架构选择的因果链

```
SA-Flow (纯Conv, 弱风格)
  ↓ 加OT → 改善trajectory
  ↓ 加CrossAttn → 但1-token=无用
  ↓ 加多token → 但共享Controller=Collapse
  ↓ 独立per-layer AdaGN → 稳定但弱
  ↓ 
LGT-X → C-G-W (去掉StyleAdaptiveSkip → 退步)
  ↓
  ↓ 回到CrossAttn: 64-token + sharpen → 0.72 (突破!)
  ↓ 但IN杀attention → twin-norm fix
  ↓ 
Cycle-NCE (0.72天花板)
  ↓ 加SB时间条件 → 但heuristic loss膨胀
  ↓ Black-dot → 清理 → 3项核心loss
  ↓
Distinct5 LANCET (0.701天花板, 架构瓶颈)
  ↓
  ↓ 诊断616/618/619 → 5个致命缺陷 → Golden Path
  ↓
620 Spatial Bridge (DINO + CrossAttn + AdaLN)
  ↓ WFI问题: Endpoint Shrinkage
  ↓ FiLM endpoint → WFI 0.39 (通过门但不稳定)
  ↓ 44+实验 clip_style 0.699-0.707 (极窄)
  ↓
  ??? 下一步: Text条件? 更强注入? End-to-end训练?
```

---

## "如果重新开始"的检查清单

1. ✅ **多token cross-attention** — 从第一天就用64+ tokens
2. ✅ **Sharpen scale** — 防止soft attention
3. ✅ **Zero-init gate** — 渐进学习
4. ✅ **MSE代替对抗loss** — latent space稳定
5. ✅ **SWD唯一style loss** — 删掉所有heuristic
6. ✅ **Kinetic energy** — 必须的稳定器
7. ✅ **DINO style encoder** — open-set是正确方向
8. ✅ **FiLM endpoint** — 唯一通过WFI<0.40的方案
9. ❌ **不要用InstanceNorm** — 在attention路径上
10. ❌ **不要用共享style controller** — Layer Collapse
11. ❌ **不要加NCE/repulsive/cycle** — 无收益或负收益
12. ❌ **不要用大batch** — micro batch更好
13. ❌ **不要训练太久** — 白化恶化
14. ❌ **不要用OT in Euclidean space** — 需要结构指纹
15. ⚠️ **clip_style不能单独用** — 必须加WFI/LPIPS
16. ❌ **不要期望Text条件在gate低时有效** — 先解决注入
17. ❌ **不要靠增加模型容量突破** — 容量不是瓶颈
18. ❌ **不要用Diff-Gram/Gram/Moment做style loss** — 完全失败
19. ✅ **Domain style >> Instance style** — 用5.77×效率的Domain表示
20. ✅ **先检查不训练的ODE质量** — Fiber-SDE=0.711, 可能不需要训练

---

## 6个月最大教训：保守偏好是统一根因

所有失败模式都可以追溯到同一个根因：**模型倾向于保守策略**。

| 时间 | 表现 | 根因 | 解决了吗? |
|------|------|------|----------|
| 01月 | 1-token attn → softmax=1.0 | 无选择性=保守 | ✅ 多token |
| 02月 | Diff-Gram style_swd=0.0 | Gram无法激活style | ✅ 放弃Gram |
| 02月 | Gram-Moment +5.5GB | 计算不可行 | ✅ 放弃 |
| 03月 | C-G-W去掉Skip → 退步 | 去掉style通路=保守 | ✅ 恢复 |
| 04月 | IN杀attention → 均匀 | 白化=保守 | ✅ 去IN |
| 05月 | Heuristic loss → black-dot | 过度约束=保守 | ✅ 清理 |
| 06月 | Gate Collapse → 0.048 | 少注入=保守 | ❌ **未解决** |
| 06月 | Endpoint Shrinkage → 0.163 | 短路径=保守 | ❌ **未解决** |
| 06月 | WFI随训练恶化 | 白化=保守 | ❌ **未解决** |

**核心洞察**: 每次我们解决了一个保守问题，模型就在另一个维度上重新选择保守。这不是偶然的——**在当前的training objective下，保守策略确实是loss最优的**。

**下一步应该做什么**: 不是继续调架构/loss，而是**改变training objective使"大胆注入style"成为loss最优策略**。
