# SchrodingerBridge Documentation

**最后更新**: 2026-07-03 (M27, Deli_AutoResearch rewrite task)
**文档结构**: 现代文档树 + 历史归档分离

---

## 1. 主线文档结构

| 目录 | 用途 | 内容简述 |
|------|------|---------|
| [baseline/](baseline/) | 相关工作 | 12 baseline 完整收敛证据（每个 baseline 独立 .md） |
| [exp/](exp/) | 我们模型的分阶段实验 | 远程+本地实验清单、实验脉络审计、ckpt 删除日志、数据集分类索引 |
| [math/](math/) | FC-SB 完整理论框架 | 12 节：问题定义、Schrödinger Bridge、Haar DWT、频域解耦、Style Conditioning、DWT Route、Stochastic DWT、EOTA、ODE 求解器等 |
| [tools/](tools/) | 工程参考手册 | 数据库分类、评估协议、基础设施、调用命令、实验经验（15 条） |
| [72/](72/) | 论文草稿 | 01-07 章节（codebase/theory/experiments/design/conclusions/cleanup/related_works）+ Pareto 散点图 |
| [archive/](archive/) | 历史文档归档 | 26 个历史目录 + 26 个根级历史文件，仅作历史追溯 |

---

## 2. 关键文档速查

### 2.1 SOTA 与 Baseline 速查

- **主线**: [exp/README.md](exp/README.md) §3 — SOTA 与 Baseline 速查表（v5 SaMam 数据对齐）
- **Baseline 完整核查**: [baseline/README.md](baseline/README.md) — 12 baseline 完整收敛证据
- **SaMam 数据完整性**: [exp/samam_data_integrity_audit.md](exp/samam_data_integrity_audit.md) — 81 checkpoint 完整曲线

### 2.2 实验脉络

- **总入口**: [exp/README.md](exp/README.md) — 整理项目总入口 + 导航
- **实验审计**: [exp/experiment_audit.md](exp/experiment_audit.md) — 5 个必须保留 ckpt + 删除建议（M23 产出）
- **远程实验**: [exp/remote_experiments.md](exp/remote_experiments.md) — 远程 I 盘所有实验清单
- **本地实验**: [exp/local_experiments.md](exp/local_experiments.md) — 本地 G 盘所有实验清单
- **Ckpt 删除日志**: [exp/ckpt_deletion_log_m26.md](exp/ckpt_deletion_log_m26.md) — 41 个无意义 ckpt 删除记录

### 2.3 数据集分类

每个实验分清数据集（256 / 5×3600 / wikiarts_5 / distinct5），分别存放：

| 数据集 | 索引文档 | 用途 |
|--------|---------|------|
| distinct5 | [../../exp/distinct5/README.md](../../exp/distinct5/README.md) | 主线论文用（5 风格 × 750 pairs） |
| wikiarts5 | [../../exp/wikiarts5/README.md](../../exp/wikiarts5/README.md) | 历史实验（非主线） |
| fewshot6 | [../../exp/fewshot6/README.md](../../exp/fewshot6/README.md) | Few-shot Pop_Art 注入实验 |
| 256 | [../../exp/256/README.md](../../exp/256/README.md) | 256 分辨率历史实验 |

### 2.4 理论与工程

- **理论**: [math/README.md](math/README.md) — FC-SB 完整理论框架
- **工程**: [tools/README.md](tools/README.md) — 数据库/评估协议/infra/调用命令/经验

### 2.5 论文草稿

| 章节 | 文档 |
|------|------|
| 总入口 | [72/README.md](72/README.md) |
| 代码库 | [72/01_codebase.md](72/01_codebase.md) |
| 理论 | [72/02_theory.md](72/02_theory.md) |
| 实验 | [72/03_experiments.md](72/03_experiments.md) |
| 设计思路 | [72/04_design_ideas.md](72/04_design_ideas.md) |
| 结论 | [72/05_conclusions.md](72/05_conclusions.md) |
| 清理笔记 | [72/06_cleanup_notes.md](72/06_cleanup_notes.md) |
| 相关工作 | [72/07_related_works.md](72/07_related_works.md) |

---

## 3. 数据真相源（Truth Source）

所有文档中的数值必须与以下真相源对齐：

| 数据源 | 路径 | 说明 |
|--------|------|------|
| 12 baseline 统一评估结果 | `exp/exp_baselines/baseline_v2/eval/unified_results.json` | SaMam 真实值 0.5816/0.2434@step20000（v4 的 0.7175/0.2423 是编造值，已修正） |
| 远程实验清单 | `docs/exp/remote_experiments.md` | M3 产出 |
| 本地实验清单 | `docs/exp/local_experiments.md` | M7 产出 |
| 实验脉络审计 | `docs/exp/experiment_audit.md` | M23 产出 |
| Baseline 完整核查 | `docs/baseline/README.md` | M22 产出 |

---

## 4. 历史归档

所有历史文档已归档到 [archive/](archive/)，包括：

- **12 个日期型历史目录**: 612-lookback, 612-phase2, 616, 618, 619, 620, 622, 625, 625_fc_sb, 627, 628, 630
- **14 个主题型历史目录**: Related, cleanup, experiments, logs, maths, model, plan, plans, presentations, references, repro_report_zh, reviews, theory, timing
- **26 个根级历史文件**: aaai2027_working_index, ablation_log, architecture, attn, bridge, cleanup_report, CLEAN_BASE*, dump*, exp_*, inmortal, known_issues, plan-612, quickstart, remote_server, review, tokenizer, writing 等

详见 [archive/README.md](archive/README.md)。

**归档文档不再维护**，仅作历史追溯用途。所有当前结论以主线文档为准。

---

## 5. 文档维护原则

1. **新增实验**: 放入对应 `exp/exp_ours/{early|phase4|local_t}/`，更新 `docs/exp/local_experiments.md` 或 `remote_experiments.md`
2. **新增 baseline**: 更新 `unified_results.json` + `docs/baseline/README.md` + `docs/72/07_related_works.md`
3. **数值变更**: 任何数值变更需在所有主线文档同步更新
4. **旧文档归档**: 不再维护的文档移到 `docs/archive/`，git commit 保留历史
5. **无效代码删除**: 确认无效后直接删除（不 ablate），git commit + 详细文档

---

**文档结构维护**: Deli_AutoResearch rewrite task (M22-M27)
**数据对齐状态**: ✅ 所有主线文档已与 `unified_results.json` 真实实验数据对齐 (v5 SaMam)
