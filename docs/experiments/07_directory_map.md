# 07 目录状态图

状态：**canonical**（当前代码/数据的权威位置）· **paper**（论文直接使用）· **raw**（原始结果，尚未迁移）· **history**（历史，只读）· **stale**（内容过期）· **scratch**（临时/可删）

## 顶层

| 路径 | 状态 | 说明 |
|---|---|---|
| `WEAVE/` | canonical | 当前方法代码、配置、评测、论文 |
| `SchrodingerBridge/` | history + **raw** | WEAVE 改名前的工作区；`experiments/rebuttal_20260716/`、`rebuttal_exps/` 是 rebuttal 原始数据；`aaai2027_v4/` 是旧副本；`aaai_submission/` 是 AAAI 2026 稿 |
| `Related_Works/` | history | 旧协议 baseline（run_511、protocol_a_800）；`repos/` 为第三方代码；5 个空 submodule |
| `Cycle-NCE/` | history | 2–4 月主线；入口 `ARCHAEOLOGY_FINAL_CN.md`、`History_Report.md` |
| `latent_cyclegan/`、`final_works/` | history | 3 月 |
| `lambda_grid/`、`step_count_sweep/`、`efficiency/`、`review_additional_experiments_aggregates/` | history | SB 方法的 review 补充实验（2026-07-14） |
| `o20_d3/` | history | 2026-07-14 的一个 D3 配置 run，用途未记录 |
| `fast_infer_ablate43/` | scratch | Cycle-NCE 模型的 ONNX/TRT 推理工具 |
| `docs/` | canonical（`docs/experiments/`）/ history（`620/`、`621/`、`results/`） | 本索引所在 |
| `results/` | stale | 旧主表与 AAAI v4 表格 bundle manifest |
| `style_data/` | canonical 数据 | P2A-256 数据 |
| `Plan_Docs/`、`spec/`、`PAPER_REVISION_PLAN.md` | stale | 计划文档 |
| `PaperOrchestra-0.2.0/` | scratch | 论文写作工具包（第三方） |
| root `*.csv`、`*.json`、`*.txt` | stale | 见 05 "root 级旧数据库" |
| root `nomachine_*.deb`、`page*.png`、`__tmp_*.py`、`session-*.md` | scratch | 可删 |

## WEAVE/

| 路径 | 状态 | 说明 |
|---|---|---|
| `config.json`、`inference.json`、`run.py`、`model.py`、`flow.py`、`trainer.py`、`wavelet.py`、`internal_dynamics.py` | canonical | 训练/推理；论文模型需叠加 `experiments/architecture/` overlay |
| `experiments/architecture/` | canonical | `hf_oriented_nohh.json` → `hf_oriented_internal_probe.json` → `hf_oriented_internal_early_stop.json`（论文 D5 模型） |
| `experiments/endpoint_adain/`、`experiments/robustness/` | canonical | AdaIN scale 轴、早停稳健性配置 |
| `experiments/inference_speed/` | scratch | 推理加速支线 |
| `scripts/batch_eval_all.py`、`scripts/run_submission_repro.ps1` | canonical | 逐 epoch 评测、复现 |
| `utils/compute_dino_metrics.py`、`utils/run_evaluation.py` | canonical | 指标 |
| `state/dino/` | paper | 全部方法 DINO sidecar |
| `exp_baselines/{aespa,stytr2}/` | paper | 两个 baseline 的 per-pair CLIP/LPIPS |
| `docs/reproduction/` | paper + raw | 复现证据账本（早停、AdaIN 轴、TGT 敏感性、ArtFID 审计） |
| `docs/model_probe/` | paper（supplement） | HF 路由探针，base checkpoint + AdaIN 1.5 |
| `docs/713/` | history | 7/13–7/15 handoff；`SUBMISSION_HANDOFF_2026-07-15.md`、`HF_ARCHITECTURE_PROBE_2026-07-13.md` 仍有参考价值 |
| `docs/79/` | canonical（定义）/ stale（数字） | 评测板定义 |
| `docs/baseline/README.md` | paper（D5 CLIP/LPIPS） | — |
| `docs/archive/`（~2290 文件） | history | 按日期快照；索引 `docs/archive/README.md` |
| `docs/delivery/`、`docs/plans/`、`docs/experiments/`、`docs/latent_migration/`、`docs/dino_s_break/`、`docs/exp/` | stale / history | — |
| `aaai2027_v4/` | paper（冻结） | AAAI 2027 投稿稿 + supplement + fig_data |
| `icme2027/` | paper（当前） | ICME 版：`weave.tex`、`weave_supp.tex`、`data/main_table.csv` |
| `aaai2027/`、`aaai2027_v4_backup_*`、`aaai_submission/`、`AuthorKit27/` | history | 旧稿与模板 |
| `results/`、`exp/ablation_v2/`、`logs/`、`EXPERIMENT_LOG.md`、`goal.md`、`history.md` | history | 旧协议/旧架构 |
| `archives/`、`tools/`（大部分） | history | 旧脚本；`tools/*other5*` 为 Other5 工具 |
