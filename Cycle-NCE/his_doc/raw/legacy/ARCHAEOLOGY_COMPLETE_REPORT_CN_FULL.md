# Latent AdaCUT 项目考古完整报告

> **生成日期**: 2026-04-03 16:15 CST
> **覆盖范围**: 2026-01-13 (DiT 初探) -> 2026-04-02 (微批革命)，共 81 天
> **数据来源**: Git 历史 186 commits + Y:\experiments 231 目录 + C:\Users\xy\full_history
> **状态**: 100% 覆盖，所有 231 个实验目录已检查，150+ 有数据实验已完整分析
> **文档规模**: 本报告统合了 5 个英文源文档 (405KB, 5,000+ 行) 的核心内容

---

## 目录

- [1. 核心执行摘要](#1-核心执行摘要)
- [2. 完整项目时间线 (81 天, 186 commits)](#2-完整项目时间线)
- [3. 代码架构深度演化](#3-代码架构深度演化)
- [4. Loss 系统完整演化](#4-loss-系统完整演化)
- [5. 全量实验数据大盘 (150+ 实验)](#5-全量实验数据大盘)
- [6. 架构消融实验详解](#6-架构消融实验详解)
- [7. 失败案例库](#7-失败案例库)
- [8. 评估管线与工具链](#8-评估管线与工具链)
- [9. 历史配置分析 (109 个配置档)](#9-历史配置分析)
- [10. 完整 Commit 日志表](#10-完整-commit-日志表)
- [11. 完整实验排行榜](#11-完整实验排行榜)
- [12. 架构关键决策总结](#12-架构关键决策总结)

---

## 1. 核心执行摘要

### 1.1 性能排行榜 Top 20

| 排名 | 实验 | 风格得分 | 内容得分 | 训练周期 | 系列 |
|:---:|:---|:---:|:---:|:---:|---|
| 1 | 1swd-dit-2style | **0.820** | 0.923 | - | SWD8 DiT (隐藏王者) |
| 2 | strong-DiT-5style | **0.813** | 0.901 | - | SWD8 DiT |
| 3 | 1swd-dit-5style | **0.812** | 0.893 | 200ep | SWD8 DiT |
| 4 | style_oa_3 (wc5) | **0.724** | 0.750 | 120ep | Attention/LPIPS |
| 5 | style_oa_8 | **0.724** | 0.742 | 60ep | Attention |
| 6 | weight_exp8 | **0.712** | 0.669 | 60ep | Weight 消融 |
| 7 | swd8_32x32 | **0.716** | - | - | SWD8 |
| 8 | exp_S1_zero_id | **0.709** | 0.671 | 150ep | Exp 策略 |
| 9 | exp_A3 | **0.691** | 0.640 | 30ep | Exp 消融 |
| 10 | style_oa_5 (120ep) | **0.698** | 0.689 | 120ep | Attention |
| 11 | decoder-H-MSCTM | 0.696 | 0.617 | 40ep | Decoder-H |
| 12 | color_01_adain_wc2 | **0.695** | 0.847 | 60ep | Color 锚定 |
| 13 | nce-gate_norm-swd_0.45 | 0.669 | 0.652 | 120ep | NCE+Gate |
| 14 | micro02_macro_patch | 0.687 | 0.714 | 80ep | Micro-Batch |
| 15 | LCE0 (5style) | **0.693** | 0.760 | 150ep | LCE |
| 16 | scan05_mid | **0.707** | 0.637 | 80ep | Scan |
| 17 | 1-decoder-patch5-15 | 0.670 | **0.790** | 15ep | Decoder Patch |
| 18 | micro05_id_anchor | 0.681 | **0.732** | 80ep | Micro |
| 19 | clocor1_id0.3 | 0.688 | 0.715 | - | clocor1 |
| 20 | master_sweep_07 | 0.666 | - | 100ep | Sweep |

### 1.2 三大关键转折点

**转折点 1: SWD 取代 Gram (2026-02-13 -> 02-21)**
- 背景: Gram Matrix 是项目早期的核心风格匹配机制 (w_stroke_gram: 80.0)
- 尝试: Diff-Gram, Moment Matching, White Gram — 全部失败
- 证据: commit 873f271 (Feb 21) "diff-gram在sdxl-fp32上表现极差"
- 结果: SWD (Sliced Wasserstein Distance) 引入后 Domain Ratio 从 1.15x 飙升到 5.77x

**转折点 2: Attention 注入失败与回归 (2026-03-26 -> 04-02)**
- 注入: CrossAttn (+241行) -> Global/Window Attention (+300行)
- 巅峰: model.py 达到 1,517 行 (11 个类, 含 Transformers)
- 问题: 8GB 显存限制下，模型退化倾向恒等映射
- 结局: 4月2日移除复杂结构，Trainer 精简 65%

**转折点 3: 微批训练革命 (2026-04-02)**
- 旧 Trainer: 1,536 行 (含 Teacher-Student 蒸馏, NCE, Classifier)
- 新 Trainer: 531 行 (-65%, 纯 SWD+Color+Identity 微批循环)
- 验证: micro_E01 (Style=0.693, Content=0.707) 性能不降反升

### 1.3 五条架构铁律 (来自消融实验)

| 消融项 | Style 变化 | Content 变化 | Direction 变化 | 结论 |
|---|---|---|---|---|
| **残差连接** | **-8.0%** | +26.1% | **-49.3%** | 残差是风格传输的唯一生命线! |
| **高频 SWD** | -4.0% | +13.5% | -21.5% | 最大风格驱动 Loss |
| **Naive Skip** | -1.1% | -8.4% | +4.0% | Skip 泄漏严重，过滤器不可少 |
| **标准 GN** | -2.8% | +6.4% | -11.5% | AdaGN 贡献温和但稳定 |
| **TV Loss** | -2.2% | +3.5% | -7.1% | 温和贡献，但可丢弃 (多次验证) |
| **TextureDict Rank** | 0.714 vs 0.716 | - | - | Rank-1 和 Rank-8 几乎一样! 纹理是极低维的 |

