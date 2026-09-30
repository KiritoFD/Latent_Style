# Task: 提升 DINO-S 到 0.5

## Goal
将 DINO-S 从 S0 baseline 0.4421 提升到 ≥0.5000，同时保持 DINO-C ≥ 0.65, CLIP-S ≥ 0.73, LPIPS ≤ 0.48。

## Milestones
1. 推理侧 WCT 探索 (Phase S1) — 完成, 天花板 0.4614
2. 训练侧 matched retraining (Phase S3) — 完成, 无效
3. 架构结构性改变 (Phase S4) — 进行中
4. RGB 空间后处理 — 待执行
5. Infra 优化 — 进行中

## Success Criteria
- DINO-S ≥ 0.5000
- DINO-C ≥ 0.6500
- CLIP-S ≥ 0.7300
- LPIPS ≤ 0.4800

## Constraints
- 显存 ≤ 11.2G (训练), ≤ 7G (评估)
- D5 数据集
- 远程 RTX 3060 12GB
- 训练禁用 DINO/CLIP 预训练模型
