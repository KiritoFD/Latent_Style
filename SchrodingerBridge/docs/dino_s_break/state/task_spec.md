# Task: Break DINO-S 0.485 Ceiling

## Goal
将 WEAVE 的 DINO-S 从当前 0.4794 (D5-512) 提升到 **0.485 以上**，同时保持结构优势：
- LPIPS ≤ 0.32 (当前 0.3111)
- DINO-C ≥ 0.78 (当前 0.7899)
- CLIP-S ≥ 0.715 (当前 0.7217)

## Background
Stage7-16 共 10 个实验确认 DINO-S 0.48 是 SAT+908K+5ep+D5 的 fundamental limit。
所有变体（增量分支/CFG/训练时AdaIN/LL部分风格化/HF WCT/Huber loss）收敛到 0.480±0.003。

根因：LL 子带携带 DINOv2 敏感的色彩/对比度统计，但被 SAT 结构性锁死。

## Success Criteria
1. DINO-S ≥ 0.485 (D5-512, 750 test pairs)
2. LPIPS ≤ 0.32
3. DINO-C ≥ 0.78
4. 训练显存 ≤ 11.2GB
5. 评估显存 ≤ 7GB

## Milestones
- M1: 识别 3+ 个结构性突破方向（不同于 Stage7-16 已试过的）
- M2: 远程训练 + 本地评估第一轮实验
- M3: 如未达 0.485，pivot 方向并启动第二轮
- M4: 达标后更新雷达图

## Constraints (from project_memory)
- 远程 RTX 3060 12GB via SSH
- 训练 Patience=2, max=10, 至少 5 epochs
- 数据集 D5 (非 Twenty-style)
- 不引入 DINO/CLIP 等外部预训练模型（避免先验污染）
- 评估 batch_size=2, full_eval_batch_size=2, ref_feature_batch_size=2
- DataLoader: num_workers=0, pin_memory=False, persistent_workers=False
