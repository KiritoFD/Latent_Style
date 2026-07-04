# 术语映射表

## 方法变体命名

| 内部代号 | 规范学术表达 | 说明 |
|---------|------------|------|
| LBM-K | LBM with kinetic regularization only | 仅使用动力学正则化的基础变体 |
| LBM-Knee | LBM with balanced regularization | 平衡正则化的推荐操作点 |
| LBM-PS | LBM with enhanced style pressure | 增强风格压力的变体 |
| LBM-PS-v2 | LBM with maximum style pressure | 最大风格压力的变体 |

## 内部检查点命名

| 内部代号 | 规范学术表达 | 说明 |
|---------|------------|------|
| I7 checkpoint | the base LBM checkpoint | 基础 LBM 检查点 |
| U4 (α=0.1) | style extrapolation with α=0.1 | 风格外推，α=0.1 |
| V6 (k=32) | patchwise AdaIN with kernel size 32 | 分块 AdaIN，核大小 32 |
| V3 (k=16) | patchwise AdaIN with kernel size 16 | 分块 AdaIN，核大小 16 |

## 技术术语规范化

| 内部术语 | 规范学术表达 | 首次出现时定义 |
|---------|------------|--------------|
| OMF | Optimal Motion Field | 最优运动场，用于学习潜在空间传输 |
| SA-SWD | Semantic-Aligned Sliced Wasserstein Distance | 语义对齐的分片 Wasserstein 距离 |
| FC-SB | Fiber-Constrained Schrödinger Bridge | 纤维约束的薛定谔桥 |
| tw-ArtFID | target-weighted ArtFID | 目标加权的艺术 FID |
| EdgePurity | Edge Purity | 边缘纯度，衡量结构保持 |
| NonCLIPAcc | Non-CLIP Style Accuracy | 非 CLIP 风格准确率 |
| EC | Efficiency-Content score | 效率-内容综合评分 |

## 工具特定词汇移除

| 内部词汇 | 替代方案 | 说明 |
|---------|---------|------|
| live-dashboard | (移除) | 不提及内部监控界面 |
| HTML payload | (移除) | 不提及内部数据格式 |
| pairing cache | precomputed endpoint set | 预计算的端点集合 |
| successor family | enhanced model variants | 增强的模型变体 |
| Stokes coefficient | style pressure coefficient | 风格压力系数 |
| barycentric targets | weighted target combinations | 加权目标组合 |
| top-2 anchor | primary anchor point | 主要锚点 |
| top-8 | extended neighborhood | 扩展邻域 |
| dual-target supervision | multi-target regularization | 多目标正则化 |
| anisotropic control | directional regularization | 方向正则化 |

## 表格列名规范化

| 原始列名 | 规范列名 |
|---------|---------|
| CLIP-S_tr | Style Score (CLIP-S) |
| 1-LPIPS_tr | Content Preservation (1-LPIPS) |
| Δ_idt,tr | Style Gain over IDT |
| tw-ArtFID_all | ArtFID (target-pooled) |
| EdgePurity | Edge Purity |
| NonCLIPAcc | Non-CLIP Style Accuracy |
