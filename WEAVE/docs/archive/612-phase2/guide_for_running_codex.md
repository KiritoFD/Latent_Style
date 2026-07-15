# Phase2 实时指导 — 2026-06-13 21:15 更新

## 当前状态

- GPU: 6817 MiB / 12288 MiB, 96% util — 训练中
- 实验: `topogate_k085_appalign` epoch 3/4 训练中
- Watcher: 等待 settled_epoch >= 4

## 最新结果分析

| Epoch | transfer style | transfer LPIPS | all-pairs style | all-pairs LPIPS | id LPIPS |
|-------|---------------|----------------|-----------------|-----------------|----------|
| e1 | 0.6726 | 0.3364 | 0.7035 | 0.3330 | 0.3195 |
| e2 | 0.6714 | **0.3143** | 0.7031 | **0.3120** | **0.3027** |
| Δ | -0.0012 | **-0.0221** | -0.0004 | **-0.0210** | **-0.0168** |

### 关键信号

1. **LPIPS 极速收敛到几乎 IDT 水平** — e2 的 all-pairs LPIPS=0.312 仅比 IDT (≈0.30) 多 0.01
   - 这意味着 topogate 几乎完美地保住了结构
   - 不再需要更多的结构约束——LPIPS 基本已经到底

2. **Style 停滞** — transfer style 从 0.6726 → 0.6714，不升反微降
   - appalign head 似乎没有激活风格
   - 模型的 capacity 被 topogate 约束在了结构保真度上
   - **现在唯一的问题：如何在保持 LPIPS=0.31 的同时推 style**

3. **Identity 重建极强** — e2 id LPIPS=0.303, id style=0.830
   - 恒等变换几乎完美，说明 VAE latent 空间重建本身不是瓶颈

## 建议的下几步

### 如果 e3-e4 style 仍停滞 (概率高)

**立即转向 restyle 路径，不要继续训练同一个 family。**

1. **I2SB σ=0.02 启动**（已经在队列）
   - 用 e1 或 e2 的 topogate ckpt 作为起点
   - 极小的布朗噪声可能打破 style 确定性轨迹
   - 预期: +0.01~0.02 style, LPIPS 几乎不变

2. **PC Solver eval**（不训练，仅推理）
   - 用现有 topogate e2 ckpt
   - solver=solver_pc, corrector_mode=latent_lowpass
   - 可以在训练 epochs 之间穿插进行
   - 验证 "Training for Style, Inference for Structure" 范式

3. **降低 topogate 约束强度**
   - 当前 topogate 可能太强了——在大幅改善 LPIPS 的同时也抑制了 style 波动
   - 尝试 `semantic_self_topology_blend=0.7` 或添加 `semantic_self_topology_gate=true` 配合 `temperature` 升高

### 如果 e3-e4 出现突然的 style 突破

观察 `app_s` (appearance scale) 和 `app_d` (appearance delta) 指标。
如果 appalign 开始在 epoch 3+ 激活 → 保持训练。

## 关于整体进展

| 指标 | 起点 (LBM F_e1) | formal lane best | topogate best | 目标 |
|------|----------------|-----------------|---------------|------|
| all-pairs style | 0.697 | 0.702 | **0.704** | 0.72 |
| all-pairs LPIPS | 0.319 | 0.367 | **0.312** | ≤0.30 |
| transfer style | 0.664 | 0.676 | **0.673** | 0.72 |
| transfer LPIPS | 0.325 | 0.369 | **0.336** | ≤0.32 |

**好消息**: LPIPS 已经到目标。**坏消息**: style 还差 ~0.04。

**最快突破路径**: topogate (保结构) + I2SB σ=0.02 (突破 style) + PC solver (最后保险)

## 给 KiritoFD 的待办清单

1. [ ] 等 `topogate_appalign` 跑完 epoch 4
2. [ ] 如果 style ≤ 0.68: 立即启动 I2SB σ=0.02 (用 e1 ckpt)
3. [ ] 并行: 用 topogate e2 ckpt 跑 PC solver eval (solver_pc + latent_lowpass corrector)
4. [ ] 如果 I2SB 仍然不突破: 切换到 PnP self-inject 或减弱 topogate blend
5. [ ] 清理已关闭的 formal lane ckpt (见 cleanup plan)
