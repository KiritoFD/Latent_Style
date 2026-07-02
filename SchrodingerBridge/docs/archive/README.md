# docs/archive — 历史文档归档

**归档日期**: 2026-07-03 (M27, Deli_AutoResearch rewrite task)
**归档原则**: 旧文档归档到 docs/archive/ + git commit 后从主文档树移除。保留历史可追溯性，不污染主线文档结构。

---

## 1. 归档原因

下列文档在 M22-M26 重组过程中被判定为历史文档，归档原因分类：

| 类别 | 说明 |
|------|------|
| **被取代** | 内容已被 docs/baseline/、docs/exp/、docs/math/、docs/tools/ 中的新文档取代 |
| **历史快照** | 日期型目录（612/616/618/619/620/622/625/627/628/630），记录特定日期的工作状态，无后续维护价值 |
| **历史实验日志** | docs/experiments/ 下 1647 个文件，记录 2026-05 ~ 2026-06 的早期实验日志，已被 docs/exp/local_experiments.md 和 docs/exp/remote_experiments.md 取代 |
| **历史清理记录** | docs/cleanup/、CLEAN_BASE*.md、cleanup_report.md 等早期清理记录 |
| **历史工具/脚本** | dump_620.py、scan_and_dashboard.py、update_dashboard.py 等一次性脚本 |
| **历史 CSV/HTML 仪表盘** | exp_all_results*.csv、exp_dashboard*.html 等已被 unified_results.json 取代 |
| **历史理论文档** | docs/maths/、docs/theory/ 已被 docs/math/ 取代 |
| **历史审稿记录** | docs/reviews/、review.md 等早期审稿/自评记录 |

---

## 2. 归档目录清单

### 2.1 日期型历史目录（12 个）

| 目录 | 原路径 | 内容简述 |
|------|--------|---------|
| 612-lookback | docs/612-lookback/ | 6月12日回看：实验库存扫描、清理脚本、分析 |
| 612-phase2 | docs/612-phase2/ | 6月12日 phase2：FIBER_BUNDLE 设计、OT 失败分析、清理脚本（802 文件） |
| 616 | docs/616/ | 6月16日：OT 理论、模块诊断、实验计划 |
| 618 | docs/618/ | 6月18日：CSGO/SCSA/StyleGallery/StyleShot 分析、风格弱问题诊断 |
| 619 | docs/619/ | 6月19日：架构设计、实现计划、理论验证（798 文件） |
| 620 | docs/620/ | 6月20日：fog 探针、理论分析、ablation audit、OT/桥/收敛诊断 |
| 622 | docs/622/ | 6月22日：历史回看（数据/架构/损失/训练演进）、FC 文档、unified 数学模型 |
| 625 | docs/625/ | 6月25日：FC-SB round2 实验完整记录、phase3 阶段总结 |
| 625_fc_sb | docs/625_fc_sb/ | 6月25日：FC-SB 实验日志 |
| 627 | docs/627/ | 6月27日：Phase 4 最终报告 |
| 628 | docs/628/ | 6月28日：消融结论、Round1 报告 |
| 630 | docs/630/ | 6月30日：Phase 4 全流程文档（phase1-4h1、4i、mask、HANDOVER 等）+ state/ |

### 2.2 主题型历史目录（8 个）

| 目录 | 原路径 | 内容简述 |
|------|--------|---------|
| Related | docs/Related/ | 相关工作方法目录（已被 docs/baseline/README.md 取代） |
| cleanup | docs/cleanup/ | 早期清理记录（loss-pruning、paper audit、worktree triage） |
| experiments | docs/experiments/ | 早期实验日志（1647 文件，2026-05 ~ 2026-06，166.7 MB） |
| logs | docs/logs/ | 旧日志目录 |
| maths | docs/maths/ | 旧理论文档目录（已被 docs/math/ 取代） |
| model | docs/model/ | 旧模型文档 |
| plan | docs/plan/ | 旧计划目录 |
| plans | docs/plans/ | 旧计划目录 |
| presentations | docs/presentations/ | 早期演示文档 |
| references | docs/references/ | 早期参考文献 |
| repro_report_zh | docs/repro_report_zh/ | 中文复现报告 |
| reviews | docs/reviews/ | 早期审稿/自评（92 文件） |
| theory | docs/theory/ | 旧理论目录（已被 docs/math/ 取代） |
| timing | docs/timing/ | 早期时序记录 |

### 2.3 根级历史文件（26 个）

| 文件 | 原路径 | 内容简述 |
|------|--------|---------|
| aaai2027_working_index_20260602.md | docs/ | 2026-06-02 AAAI2027 工作索引 |
| ablation_log.md | docs/ | Phase 2 A/B 测试日志 |
| architecture.md | docs/ | 旧架构文档 |
| attn.md | docs/ | 注意力机制笔记 |
| bridge.md | docs/ | 桥接机制笔记 |
| cleanup_report.md | docs/ | Phase 1 清理报告 |
| CLEAN_BASE.md | docs/ | 旧 clean base 文档 |
| CLEAN_BASE_V2.md | docs/ | 旧 clean base v2 文档 |
| dump620c.py | docs/ | 620c 导出脚本 |
| dump_620.py | docs/ | 620 导出脚本 |
| scan_and_dashboard.py | docs/ | 扫描+仪表盘脚本 |
| update_dashboard.py | docs/ | 仪表盘更新脚本 |
| exp_all_results.csv | docs/ | 旧实验结果 CSV |
| exp_all_results_remote.csv | docs/ | 旧远程实验结果 CSV |
| exp_unified.csv | docs/ | 旧统一结果 CSV |
| exp_dashboard.html | docs/ | 旧仪表盘 v1 |
| exp_dashboard_v2.html | docs/ | 旧仪表盘 v2 |
| inmortal.md | docs/ | inmortal 系列笔记 |
| known_issues.md | docs/ | 旧已知问题列表 |
| plan-612.md | docs/ | 6月12日计划 |
| quickstart.md | docs/ | 旧快速开始指南 |
| remote_server.md | docs/ | 旧远程服务器指南 |
| review.md | docs/ | 旧审稿笔记 |
| tokenizer.md | docs/ | 旧 tokenizer 文档 |
| wrinting.md | docs/ | 旧写作笔记（typo） |
| writing.md | docs/ | 旧写作笔记 |

---

## 3. 保留的现代文档结构

以下文档不在本归档范围内，保留在 docs/ 根目录：

| 目录/文件 | 用途 | 创建/维护里程碑 |
|-----------|------|----------------|
| docs/baseline/ | 相关工作（12 baseline 完整收敛证据） | M22 |
| docs/exp/ | 我们模型的分阶段实验（remote + local + audit + 数据集分类） | M3/M7/M23/M25 |
| docs/math/ | FC-SB 完整理论框架（12 节） | M24 |
| docs/tools/ | 工程参考手册（数据库/评估协议/infra/调用命令/经验） | M24 |
| docs/72/ | 论文草稿（01-07 章节 + Pareto 散点图） | 持续维护 |
| docs/README.md | 文档总入口 | M27 更新 |

---

## 4. 引用归档文档的正确方式

当主线文档需要引用归档内容时，使用相对路径：

```
详见归档：[docs/archive/630/phase4d_multi_level_dwt.md](../archive/630/phase4d_multi_level_dwt.md)
```

**注意**：归档文档内容不再维护，仅作为历史追溯用途。所有当前结论以主线文档（docs/baseline/、docs/exp/、docs/math/、docs/tools/、docs/72/）为准。

---

## 5. 恢复归档文档

如需将某归档文档恢复到主线，执行：

```powershell
# 示例：恢复 docs/archive/630/phase4d_multi_level_dwt.md 到 docs/exp/
Move-Item "docs\archive\630\phase4d_multi_level_dwt.md" "docs\exp\phase4d_multi_level_dwt.md"
git add -A docs/
git commit -m "docs: restore phase4d_multi_level_dwt.md from archive"
```

---

**归档执行**: Deli_AutoResearch rewrite task, iteration 3, M27
**Git commit**: 见本归档对应的 commit message
