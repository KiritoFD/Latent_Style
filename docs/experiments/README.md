# 实验数据总索引（Experiment Index）

> 更新：2026-09-30。本目录是整个仓库实验数据的**唯一入口**。旧文档仍保留在原位置，但以本目录的判定为准；
> 若本目录与其它文档冲突，按 [02_results_of_record.md](02_results_of_record.md) 的"数据源优先级"处理。

## 一句话现状

- **当前方法**：WEAVE（Haar 小波解耦的潜空间速度匹配），代码在 `WEAVE/`。
- **当前论文**：ICME 版 `WEAVE/icme2027/weave.tex`（6 页含参考文献）+ `weave_supp.tex`；
  AAAI 2027 投稿版冻结在 `WEAVE/aaai2027_v4/`（不再修改）。
- **主表唯一数据源**：`WEAVE/icme2027/data/main_table.csv`（每个数都带出处列），
  由 `WEAVE/icme2027/tools/make_main_table.py` 生成论文 Table I。
- **消融/种子/参考池原始数据**：仍在旧目录 `SchrodingerBridge/experiments/rebuttal_20260716/`
  与 `SchrodingerBridge/rebuttal_exps/`（WEAVE 改名前的工作区，未迁移，见 [06](06_known_issues.md)）。

## 文档地图

| 文件 | 内容 | 什么时候看 |
|---|---|---|
| [01_protocol.md](01_protocol.md) | 三个评测板（D5-512 / P2A-256 / R5-WikiArt）、Other5、指标定义、IDT/TGT、评测脚本 | 跑评测、写实验设置 |
| [02_results_of_record.md](02_results_of_record.md) | 论文里每个数字 → 原始文件 → checkpoint；已验证 / 仅论文 两种状态 | 引用任何数字之前 |
| [03_ablations_robustness.md](03_ablations_robustness.md) | 消融、逐 epoch 曲线、早停 regret、敏感性扫描、参考池、ArtFID、结构指标、HF 路由探针、SD1.5 插件 | 写消融 / 答审稿 |
| [04_baselines.md](04_baselines.md) | 14 个对比方法的来源、运行方式、数据出处、时间来源；未入表的 baseline | 补 baseline、核对 Table I |
| [05_history_timeline.md](05_history_timeline.md) | SA-Flow → Cycle-NCE → Latent CycleGAN → SchrodingerBridge/LANCET → 620/621 → WEAVE 的时间线与每阶段最好结果 | 追溯历史、写 related/动机 |
| [06_known_issues.md](06_known_issues.md) | 已发现的数字冲突、过期文档、缺失原始文件、ICME 版已做的更正 | 修数据、准备 rebuttal |
| [07_directory_map.md](07_directory_map.md) | 仓库每个顶层目录的状态：canonical / 论文用 / 历史 / 废弃 | 找文件、清理仓库 |

## 常用问题速查

| 我要找 | 去哪里 |
|---|---|
| WEAVE 主结果 D5（0.4918/0.7128/0.2595/0.8102） | `SchrodingerBridge/experiments/rebuttal_20260716/expD/results.json` 的 `D0_full` |
| 主表全部方法 × 三个板 | `WEAVE/icme2027/data/main_table.csv` |
| 基线 DINO-S / DINO-C | `WEAVE/state/dino/<board>__<method>.json`（汇总：`WEAVE/aaai2027_v4/fig_data/dino_main.json`） |
| 训练 / 推理配置 | `WEAVE/config.json` + `WEAVE/experiments/architecture/hf_oriented_internal_early_stop.json`（overlay 链）+ `WEAVE/inference.json` |
| 早停内部探针轨迹 | `WEAVE/docs/reproduction/internal_dynamics_early_stop.csv`、`internal_dynamics_robustness.csv` |
| 逐 epoch 外部指标（各 seed、D3/D4/D5 消融） | `SchrodingerBridge/experiments/rebuttal_20260716/expA_*/per_epoch_metrics.csv` + `oracle_regret.json` |
| ArtFID 审计 | `SchrodingerBridge/rebuttal_exps/experiments/rebuttal_20260716/expC_canonical_artfid/canonical_artfid.json`；`WEAVE/docs/reproduction/artfid_d5_audit.md` |
| 审稿视角的证据审计（中文） | `SchrodingerBridge/rebuttal_exps/docs/reviewer_audit_and_required_experiments.md` |
| 旧协议（CLIP/LPIPS only）的大表 | `experiment_database_all.csv`（2503 行，截止 2026-06-15，已冻结，不可与 DINO 指标混用） |

## 维护规则

1. 新实验必须记录：git commit、resolved config（及 sha256）、checkpoint 路径、评测 manifest、seed、选中 epoch、原始 JSON/CSV 路径。
2. 论文数字只能来自 `main_table.csv` 或本目录 [02](02_results_of_record.md) 列出的原始文件；**不要**再从旧 md 手抄。
3. 新结果进入仓库后，在 [02](02_results_of_record.md) 登记一行，状态写 `verified`（有原始文件）或 `paper-only`（只在论文/笔记中）。
4. 不再在 `SchrodingerBridge/` 下新增内容；新的原始结果放到 `WEAVE/docs/reproduction/`。
