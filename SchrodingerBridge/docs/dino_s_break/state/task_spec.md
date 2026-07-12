# Task: WEAVE Radar Chart Dominance (Round 5+)

## Goal
通过架构解放（改架构/loss/扩容）让 WEAVE 在雷达图对比中胜出。
第一目标：DINO-S（突破 0.49）；第二目标：CLIP-S（恢复 ≥0.72）；内容保持合理领先（LPIPS<0.30, DINO-C>0.80）。

## Current State (Round 4 complete)
- adain_scale sweep 完成，发现 adain_scale 是推理参数（不影响训练）
- brk_s (adain=1.6) 是平均数峰值：avg=0.6012, 4/4 Pareto 改进 brk_a
- brk_q (adain=2.0) DINO-S 最高=0.4859 但 CLIP-S=0.7075
- 所有 adain sweep 的 checkpoint 相同（训练配置一致）

## Round 5 Directions (4 architecture liberation experiments)

### Direction 1: brk_u_hh_head — HH 子带笔触监督
**数学动机**：当前 HH head 禁用，对角高频（笔触方向）无监督。启用后：
$$\mathcal{L}_{fm} = 0.3\mathcal{L}_{ll} + 1.0\mathcal{L}_{lh} + 1.0\mathcal{L}_{hl} + 2.0\mathcal{L}_{hh}$$
**配置**：enable_hh_head=true, spectral_w_hh=2.0
**预期**：DINO-S↑（笔触是风格关键维度），CLIP-S 中性

### Direction 2: brk_v_train_adain — 训练时 AdaIN 对齐
**数学动机**：endpoint_adain_scale=1.0 推理时生效，但训练 target 未经过 AdaIN，导致 train-test mismatch。
$$x_1^{*,train} = \text{AdaIN}(x_1^*, x_s, \alpha_{train}=1.0)$$
**配置**：train_adain_enabled=true, train_adain_scale=1.0
**预期**：CLIP-S↑（mismatch 消除），DINO-S 保持

### Direction 3: brk_w_hf_wct — 高频 WCT（完整协方差迁移）
**数学动机**：AdaIN 仅匹配对角协方差（mean+std），WCT 匹配完整协方差：
$$f_w = \Sigma_c^{-1/2}(f - \mu_c), \quad f_{colored} = \Sigma_s^{1/2} f_w + \mu_s$$
**配置**：hf_wct_enabled=true, hf_wct_beta=0.5
**预期**：DINO-S↑（更丰富纹理统计），CLIP-S 中性

### Direction 4: brk_x_dim96 — backbone 扩容
**数学动机**：backbone 315K（35%）表达力不足，dim 64→96 扩容到 ~710K（+125%）
**配置**：base_dim=96, batch_size=64（VRAM 安全）
**预期**：所有指标↑（更强表达力），需更长训练

## Success Criteria
1. DINO-S > 0.49（突破当前 0.4859 天花板）
2. CLIP-S > 0.72（恢复到 baseline 水平）
3. LPIPS < 0.30, DINO-C > 0.80（内容保持合理领先）
4. 雷达图上 4 指标整体胜出 baseline

## Constraints
- 远程 RTX 3060 12GB via SSH
- 训练显存 ≤ 11.2GB
- 评估显存 ≤ 7GB
- 不引入 DINO/CLIP 外部预训练模型到训练
