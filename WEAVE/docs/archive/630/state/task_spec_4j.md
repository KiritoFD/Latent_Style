# Phase 4J Task Spec: Few-shot + 结构性突破 (方案 A/B/C)

## 目标
1. 在 Phase 4I.11 SOTA (clip=0.7250, lpips=0.3129) 基础上,通过结构性改进进一步放大优势
2. 实施 few-shot style_mem 个性化实验 (Textual Inversion 范式)
3. 验证 "Style Is Learned, Not Extracted" 理论

## 里程碑
- M1: 方案 B (DWT-Routed Cross-Attention) 实施 + 评估
- M2: 方案 A (WCT-Aligned Target) 实施 + 评估
- M3: 方案 C (Progressive Alpha Scheduling) 实施 + 评估
- M4: Few-shot style_mem 优化实验 (freeze_mode=tokenizer_only + 新风格)
- M5: 综合最优组合 + 最终 SOTA 确认

## 成功标准
- 基础: 两方面超越 Phase 4I.11 SOTA (clip>0.7250, lpips<0.3129)
- 进阶: clip>0.7300, lpips<0.3050 (大幅双超越 SaMam)
- Few-shot: 新风格 style_mem 在 ≤50 步内收敛,clip>0.70

## 方向多样性约束
- 4J.1 (方案B): 频域解耦 cross-attention (结构)
- 4J.2 (方案A): 训练目标对齐 (数据)
- 4J.3 (方案C): 推理时序调度 (推理)
- 4J.4 (Few-shot): 参数高效微调 (训练范式)
四个方向互不相同,覆盖结构/数据/推理/训练四个维度。

## 约束
- 条件编译 (config 参数) 控制,不影响其他测试
- 显存 ≤ 10.8GB (训练), ≤ 7GB (评估)
- 每个方向独立训练到收敛 (Patience=2, max=10, ≥5 epochs)
- 数据集路径: I:/wikiart_distinct5_samam_512_latents_ema
