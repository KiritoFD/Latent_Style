# Rebuttal Supplementary Experiments

## Goal
堵住 Theory 和 Methodology 方向 Reviewer 的嘴，补充 7 个实验。

## Milestones
1. Exp2: IDT-TGT方差分析 (纯评估)
2. Exp1c: AdaIN scale推理敏感度 (纯推理)
3. Exp1a: λ_LL训练敏感度 (5配置×6epoch)
4. Exp1b: α训练敏感度 (5配置×6epoch)
5. Exp3: 梯度门控鲁棒性 (9组训练)
6. Exp4: 多尺度小波消融
7. Exp5: 极端高频压力测试
8. Exp6: 新Baseline核实

## Success Criteria
- Exp1: 3张折线图，证明参数在合理区间内稳定
- Exp2: 带误差棒的Sandwich区间图，σ < 0.015
- Exp3: 3×3表格，门控稳定在epoch 3-4
- Exp4: 对比表格，含训练时间增量
- Exp5: 定性可视化图
- Exp6: 代码可用性报告

## Key Paths
- Checkpoint: runs/submission/hf_oriented_internal_early_stop/epoch_0004.pt
- Generated images: exp/repro_weave_d5/
- Remote: ssh -p 2222 administrator@100.115.18.62
- WEAVE root (remote): I:\Github\Latent_Style\WEAVE
