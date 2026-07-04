# 621 深度排查与消融实验报告

> 建立日期: 2026-06-21  
> 目标: 穷尽当前模型所有实现，建立白化/雾化定量指标，深入确认原因机理，同步数学理论

---

## 文档索引

| 文件 | 内容 |
|------|------|
| `README.md` | 本文件，总索引 |
| `architecture_audit.md` | 完整模型架构审计（620 + legacy） |
| `experiment_inventory.md` | 全分支实验清单与消融结果汇总 |
| `whitening_metrics.md` | 白化/雾化定量指标体系（图像空间 + 潜空间） |
| `probe_design.md` | 内部探针设计方案与诊断方法论 |
| `ablation_results.md` | 消融实验结果（含开销分析） |
| `theory/whitening_mechanism.md` | 白化机制数学理论 |
| `theory/endpoint_shrinkage.md` | Endpoint收缩的严格数学分析 |
| `theory/norm_collapse.md` | 归一化统计塌缩理论 |
| `theory/attention_bottleneck.md` | Cross-Attention信息瓶颈理论 |
| `decisions.md` | 决策台账：哪些有用/哪些该删 |
| `remote_3060_setup.md` | 远程3060 WSL环境与实验配置 |
| `next_steps.md` | 下一步行动计划 |
| `historical_archaeology.md` | **历史数据考古综合分析** (22K CSV, 15分支, 4阶段时间线, 统一数学理论) |
| `csv_analysis_report.md` | CSV统计分析报告 (自动生成) |

---

## 项目概览

### 核心问题
620模型生成的风格迁移图片呈现系统性白化/雾化（whitening/fogging）：
- 低对比度、低饱和度、高亮度
- 视觉上像蒙了一层白雾
- 同时影响 identity 和 style_transfer

### 当前状态
- **最优**: clip_style=0.7051, WFI=0.3906（刚过0.40放行门）
- **目标**: clip_style≥0.72, WFI≤0.20（接近Seedream IDT水平0.158）
- **根因已确认**: Endpoint shrinkage + style→endpoint FiLM容量不足

### 两代模型架构
1. **Legacy LANCET** (`src/model.py`, `src/lancet_backbone.py`): U-Net encoder-decoder, 多种style tokenizer, OT coupling
2. **620 SpatialBridge** (`src/model620.py`): 纯transformer blocks, DINO cross-attention, velocity/endpoint prediction

### 关键发现
- 白化起源于 `predict_endpoint(t=0)`，不是solver
- Endpoint只移动了目标方向的16%（α=0.16）
- Style gate=0.05→style信号被压制→条件期望坍缩
- GroupNorm(1)压缩动态范围
- FiLM endpoint head hd512将WFI从0.49降至0.39

---

## 实验覆盖范围

### 已覆盖分支
| 分支 | 方法 | 状态 |
|------|------|------|
| codex/620-spatial-bridge | 620 SpatialBridge | 当前开发 |
| SWD | Sliced Wasserstein Distance | 早期实验 |
| Gram-Moment | Gram矩阵+矩匹配 | 结果差 |
| Diff-Gram | 微分Gram | 极差(sdxl-fp32) |
| Thermal | 热力学/LoRA | 风格好但质量差 |
| attn | 注意力优化 | 3060适配 |
| multistep-texture | 多步纹理 | CLIP-S达0.72 |
| re-SWD | 重做SWD | style-8注入 |
| Classify | 分类器信号 | 结构太强 |
| Cycle-upscale | 循环+上采样 | structure loss无用 |
| Style8_Moment+SWD | 8风格矩+SWD | Few-shot |
| sdxl-fp16 | SDXL半精度 | 差 |
| main | Tokenizer | 基线 |

### 已完成消融维度
1. SWD weight: 12/16/20
2. Velocity length: 1.0/0.2/0.04
3. Attention mode: softmax/gated/gated_raw/relu2/style_select/sparsemax
4. Endpoint head: velocity/endpoint_lowhigh + FiLM
5. FiLM hidden dim: 128/512
6. Gate init: 0.05/0.3
7. Style source: DINO patches/intrinsic(CNN)
8. Cross-attention skip: coarse layers
9. Training target: source_low_target_high/target_linear/pure_vertical_flow
