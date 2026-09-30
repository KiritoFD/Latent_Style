# Latent Style 项目考古总览

> 2026-01-13 ~ 2026-06-23 | 6个月 | 18个branch | 645+实验 | 完整数据驱动梳理

## 项目目标

在latent space实现通用风格迁移（open-set, 非固定风格数），核心挑战是**风格注入强度 vs 内容保持 vs 训练稳定性**的三难困境。

## 文档结构

| 文档 | 内容 |
|------|------|
| [01_dataset_evolution.md](./01_dataset_evolution.md) | 数据集从2-style到open-set的10次演变 |
| [02_architecture_evolution.md](./02_architecture_evolution.md) | 8个主要架构的完整演变链 |
| [03_loss_evolution.md](./03_loss_evolution.md) | Loss函数从1项到9项再回到3项的膨胀与清洗 |
| [04_training_and_results.md](./04_training_and_results.md) | 训练策略、硬件约束、关键实验结果时间线 |
| [05_lessons_learned.md](./05_lessons_learned.md) | 什么有效/什么无效/因果链/重做检查清单 |
| [06_complete_experiment_data.md](./06_complete_experiment_data.md) | **完整实验数据汇总：18 branch × 全指标** |
| [07_theory_corrections.md](./07_theory_corrections.md) | **用645+实验数据修正5个理论假设，提出4个新理论** |
| [08_predictions.md](./08_predictions.md) | **基于统一理论+620完整数据的下一步预测** |
| [09_complete_intervention_map.md](./09_complete_intervention_map.md) | **7轴×30+改动的量化影响图，每个Δ三元组** |
| [10_unified_mathematical_model.md](./10_unified_mathematical_model.md) | **四机制耦合退化吸引子模型，6命题+实验验证+信号衰减链** |
| [../../621/theory/latent_style_mathematical_theory.md](../../621/theory/latent_style_mathematical_theory.md) | **836行统一数学理论：4机制+11预测+6阶段路线** |
| [remote_eval/all620.json](./remote_eval/all620.json) | **620远程187条eval原始数据** |

## 关键数字速查

| 指标 | 最佳值 | 来源 | 备注 |
|------|--------|------|------|
| clip_style (通用, 5-style) | **0.731** | XPred_Kmanifold_Pattn_Stokes | 但LPIPS=0.618内容差 |
| clip_style (通用, 好平衡) | **0.701** | LANCET K e1 / I2SB orth e9 | LPIPS=0.36 |
| clip_style (per-style) | **0.793** | SaMST Symbolism | 不可比但参考 |
| clip_style (620远程最佳) | **0.6765** | 620_lowmix05_diag_b64 e1 | 但LPIPS=0.349差 |
| clip_style (620远程平衡) | **0.6751** | 620_lowswd_formal e2 | LPIPS=0.278, allpairs=0.708 |
| clip_style (620有WFI最佳) | **0.6735** | 620_film_formal e8 | WFI=0.509, LPIPS=0.310 |
| clip_style (620本地) | **0.666** | 620_t5base_b4 e8 | 完整eval |
| clip_style (不训练ODE) | **0.711** | Fiber-SDE σ=0.08 | **不训练比训练好** |
| LPIPS (最佳内容保持) | **0.268** | 620_swd20 | 极端内容保 |
| LPIPS (好平衡) | **0.314** | LatAff s0.35 | clip_style=0.677 |
| WFI (最低白化) | **0.391** | 620_film_v5_hd512 1ep | 不稳定 |
| WFI (620远程最低) | **0.410** | 620_film_gate03_5ep e5 | 但clip_style低 |
| Endpoint Shrinkage alpha | **0.163** | 620诊断 | 只走16%目标方向 |
| Cross-attn entropy | **6.24** | 620 | near-uniform |
| Style gate (620) | **0.048** | 620所有实验 | 几乎关闭 |
| Style-content correlation | **+0.94** | Style8_Moment+SWD | 强耦合 |
| Domain vs Instance ratio | **5.77×** | Style8_Moment+SWD | Domain远优 |
| 620远程实验数 | **39/187** | all620.json | 39个实验×多epoch=187 |
| 实验总数 | **645+** | all_experiments.csv | 含远程+本地 |
| 分支总数 | **18** | git branch -a | 含6个已合并 |

## 三难困境

```
          风格强 (clip_style↑)
         /          \
        /            \
 内容保(LPIPS↓) ── 训练稳(loss收敛)
```

**数据验证的tradeoff**:
| 方案 | 风格 | 内容 | 稳定性 |
|------|------|------|--------|
| XPred Pattn | **0.729** | 0.618 | 差 |
| LANCET K | 0.701 | 0.362 | 好 |
| 620 notext | 0.665 | **0.287** | 好 |
| Fiber-SDE(不训练) | 0.711 | 0.337 | 完美 |

## 架构代际与clip_style天花板

```
Thermal (01-02月)    ──── 0.59  ─── 物理方法，moderate
CGW sweep (03月)     ──── 0.69  ─── 8变体无差异(0.68-0.69)
attn + 64-token(03月)──── 0.72  ─── **6个月最大突破+0.03**
style_oa_8 (03月)    ──── 0.724 ─── Cycle-NCE天花板
SB cleanup (05月)    ──── 0.694 ─── 回退但更稳
LANCET K (06月)      ──── 0.701 ─── 回到0.72水平
XPred+Pattn (06月)   ──── 0.731 ─── **绝对最高**但LPIPS差
620 (06月)           ──── 0.665 ─── **回退0.04**但LPIPS好
SaMST per-style      ──── 0.760 ─── 上限参考
```

## 18个Branch全景

| Branch | 目的 | Best clip_style | Best LPIPS | 合并? | 结论 |
|--------|------|---------------|-----------|-------|------|
| **Classify** | 分类器引导训练 | 0.593 (overfit50) | 0.454 | ❌ | 分类器不能提升style |
| **Cycle-upscale** | Cycle-MSE, 分辨率注入 | ~0.59 | ~0.39 | ✅ | MSE>对抗，基线建立 |
| **Diff-Gram** | 可微Gram/黎曼 | **0.0** (全部) | 0.977 | ❌ | **完全失败** |
| **Gram-Moment** | Gram+Moment+半群 | ~0.50 | ~0.43 | ❌ | 半群+5.5GB, 不可行 |
| **SWD** | SWD损失+分类器评估 | ~0.55 | ~0.42 | ❌ | SWD边界效果 |
| **Style8_M+SWD** | 系统Cycle-NCE实验 | 0.593/0.552 | 0.386 | ✅ | Domain>Instance 5.77× |
| **Thermal** | 热力学, LoRA, proxy | 0.590 | 0.47 | ✅ | Proxy失败, LoRA有效但贵 |
| **attn** | Cross-attention, CGW | **0.721** | 0.585 | ❌ | 突破0.72但chessboard |
| **style-inj-priority** | Style-first, proto-sep | 0.467 | 0.244 | ❌ | 代码regression+style collapse |
| **multistep-texture** | 多步纹理 | ~0.59 | ~0.39 | ❌ | 仅infra改进 |
| **re-SWD** | SWD revisit, FP32 | ~0.55 | ~0.42 | ❌ | SWD在FP16=NaN |
| **sdxl-fp16** | SDXL移植 | ~0.59 | ~0.39 | ❌ | scale=0.154, 无突破 |
| **rebuild-clean** | 仓库重建 | - | - | ✅ | 清理基线 |
| **backup-pre-clean** | LANCET全量备份 | **0.731** | 0.475 | 活跃 | AAAI提交数据 |
| **620-spatial-bridge** | DINO+CrossAttn | 0.6765 | **0.268** | 当前 | **LPIPS最优但style弱, 39远程实验187条eval** |
| pushfix-clean/ff | Cherry-pick推送 | 同backup | 同backup | 活跃 | 同数据 |
| replay-ordered | 考古replay | 0.810 (overfit) | 0.319 | 活跃 | LANCET F ArtFID=122.6 |

## 核心发现：被数据推翻的5个假设 vs 新提炼的4个理论

### ❌ 被推翻

1. **"620新架构应该更好"** → clip_style回退0.04 (0.701→0.665)
2. **"Text条件提升风格"** → T5 vs no-T5差0.001
3. **"更大模型=更好"** → 64→128 dims差0.001
4. **"训练越久越好"** → WFI 0.39→0.47恶化
5. **"SWD是好style loss"** → SWD梯度与v_target正交(cos=-0.024)

### ✅ 新理论

1. **Gate Collapse** — 模型学到了"保守策略"，gate收敛到0.048
2. **Training-Output Mismatch** — 不训练ODE(0.711)比训练后(0.701)更好
3. **有效style维度极低** — 21个CGW configs和36个620消融结果几乎一样
4. **保守偏好是统一根因** — IN→均匀attn, Gate→低值, Endpoint→shrinkage, 训练→白化

## 代码目录迁移

```
Root/ (01月) → Thermal/ (01月) → Cycle-NCE/ (02月) → SchrodingerBridge/ (05月至今)
  SA-Flow         LGT-X          C-G-W               SB → Distinct5 → 620
```

## 当前状态 (2026-06-23)

- **最优clip_style**: 0.731 (XPred Pattn, 但LPIPS=0.618)
- **最优平衡**: 0.701 (LANCET K / I2SB orth, LPIPS≈0.36)
- **最优LPIPS**: 0.268 (620_swd20, 但clip_style低且白化)
- **620远程数据**: 39实验×187 eval条目已提取，4个有WFI数据
  - 最佳clip_style: 0.6765 (lowmix05_diag e1, 但LPIPS=0.349)
  - 最佳平衡: 0.6751/0.2767 (lowswd_formal e2)
  - WFI-Style tradeoff确认: film_gate03 WFI=0.41→film_formal WFI=0.51
  - 架构变体(adapter/gate12/moe)差异<0.001，无关紧要
  - SWD宽度最佳=12, NFE无影响, sigma无影响
  - 内容退化随训练普遍存在，style/content最佳比通常在ep3-4
- **620核心问题**: Gate Collapse (0.048) + Endpoint Shrinkage (alpha=0.163) + 白化随训练恶化
- **统一数学理论已建立**: docs/621/theory/latent_style_mathematical_theory.md (836行)
  - 4个耦合机制形成自强化退化吸引子
  - 模型仅保留0.5%原始style信息到endpoint
  - 11个可证伪预测 + 6阶段修复路线图
- **Text条件**: 在gate=0.048时无效，需先解决注入问题
- **645+实验, 39个620远程消融**: 无任何单维度突破

## 历史commit关键节点

| 日期 | Commit | 里程碑 |
|------|--------|--------|
| 01-13 | `c7547f456` | 项目启动, encode wikiarts |
| 01-15 | `810cdc32` | transformer fail, 回退SA-Flow |
| 01-28 | `c043767` | 加cross_attn, "风格强多了" |
| 02-10 | `9e7362b` | "Cycle改MSE是对的" |
| 02-16 | `54d120e` | "structure loss完全没用" |
| 03-29 | `60b3bfe` | 64-token cross-attention, "效果明显" |
| 04-06 | `cd8cb2b` | **IN杀注意力修复** |
| 04-07 | `60ee4c6` | "从basic重新开始" |
| 05-07 | `af9d0b2` | SB引入 |
| 05-09 | — | Black-dot危机 |
| 05-19 | 4 commits | **Phase 1 Cleanup** (losses 942→340) |
| 06-01 | — | Distinct5 LANCET训练 |
| 06-16 | `828151b` | 616 OT失败诊断 |
| 06-19 | `d94b5d4f6` | 620 Spatial Bridge主线上线 |
