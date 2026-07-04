# Task: Baseline & Our-Model Re-evaluation on 256 Dataset

**Created**: 2026-07-04 11:00 (Asia/Shanghai)
**Owner**: TRAE agent
**Framework**: Deli_AutoResearch
**Mode**: Zero-interaction, autonomous

## 1. Goal

在同一 256 数据集（与 pixel256/latent256 实验相同）上重跑主要 baseline 与我们模型，回收数据并与 512 结论对比，验证结论一致性。

## 2. Scope

### 2.1 主要 baseline（用户明确指定）
1. **SAMST** — 像素空间 baseline，已有 512 评估
2. **SaMam** — 像素空间 baseline，已有 512 评估
3. **其他 train-free 方法**：从 `docs/Related/baseline_methods_catalog.md` 中筛选（AdaIN/WCT/StyTr²/稻香 等）

### 2.2 我们的模型
- **spectral_ode 主线**（`620_spectral_v11_ll10_hh20`）：在 256 输入下评估（用 latent256 配置作代理已完成；可补充 spectral_ode 主线 ckpt 在 256 测试集上的直接评估）

### 2.3 探索性任务（可选）
- 把 SAMST / SaMam 迁移到 latent 空间：评估迁移难度，难度大则放弃并记录

## 3. Success Criteria

1. 所有主要 baseline 在同一 256 测试集上完成评估，summary.json 落盘
2. 与 512 baseline 结果对比表落盘到 `docs/baseline_256/compare_256_vs_512.md`
3. 结论一致性判断（CLIP-S 排序、LPIPS 排序是否与 512 一致）
4. SAMST/SaMam latent 迁移可行性记录（即使放弃也要写明原因）

## 4. Hard Constraints (from project_memory)

- 显存 ≤ 7G（评估时, batch_size=2）
- 数据集路径：`I:/wikiart_distinct5_samam_512_classview/test`（512 classview，pixel256/latent256 已用此测试集）
- 远程：`ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62`，RTX 3060 12GB
- 命令加 30s timeout
- 训练 Patience=2, max=10
- 所有检查点都要评估，说明复现跑到收敛
- 数据汇总要完整

## 5. Milestones

| M | 内容 | 状态 |
|---|---|---|
| M1_env_probe | 远程 256 数据集 + baseline 代码 + 依赖探测 | pending |
| M2_samst_256 | SAMST 256 评估 | pending |
| M3_samam_256 | SaMam 256 评估 | pending |
| M4_trainfree_256 | train-free 方法（AdaIN/WCT 等）256 评估 | pending |
| M5_ourmodel_256 | 我们模型 spectral_ode 256 评估（如缺） | pending |
| M6_compare | 256 vs 512 对比文档 | pending |
| M7_latent_migration | SAMST/SaMam latent 迁移探索（可选） | pending |
| M8_final_report | 最终报告 + 数据汇总 | pending |

## 6. Test Set

`I:/wikiart_distinct5_samam_512_classview/test`（5 风格 × 50 张 = 250 ref，5 × 30 = 150 src，5×5=25 对 × 30 src = 750 生成）

与 pixel256/latent256 实验完全相同的测试集，确保横向可比。

## 7. Notes

- latent256 已完成（见 `docs/exp/pixel.md`），不重跑
- pixel256 已完成（见 `docs/exp/pixel.md`），不重跑
- 远程 I 盘已有 `wikiart_distinct5_samam_512_pixel256/train`（pixel256 训练数据，5×1000 张 .pt）
- 远程 I 盘已有 `wikiart_distinct5_samam_512_latent256/train`（latent256 训练数据，5×1000 张 .pt packed）
