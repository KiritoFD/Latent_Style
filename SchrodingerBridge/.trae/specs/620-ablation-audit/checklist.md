# Checklist: 620 全面消融实验与设计审计

## Phase 0: Git 历史调研

- [x] 已列出所有相关本地与远程分支
- [x] 已提取各分支 commit message 中的实验结果
- [x] 已对比关键分支的代码差异
- [x] 已读取 `docs/620/` 中的历史结果
- [x] 已输出 `docs/620/fog/ablation_audit/git_history_digest.md`

## Phase 1: 基线整理与消融矩阵设计

- [x] 已整理 Round 1 `base_swd8`、白化修复前 `gated`、此前最优 `endpoint_film_hd512` 三条基线
- [x] 已输出 `docs/620/fog/ablation_audit/baseline_table.md`
- [x] 已列出核心维度与扩展维度
- [x] 已输出 `docs/620/fog/ablation_audit/ablation_matrix.md`
- [x] 已创建 `tools/generate_ablation_configs.py`
- [x] 已创建 `tools/run_ablation_batch.py`
- [x] 已确认 `tools/run_eval_with_wfi.py` 可在本地稳定运行

## Phase 2: 核心维度消融实验

- [x] Attention 机制消融完成（gated / softmax / gated_raw / relu2 / style_select / sparsemax）
- [x] StyleFiLM 消融完成（true / false）
- [x] Endpoint 结构消融完成（velocity / endpoint_lowhigh × film on/off × hd128/256/512）
- [x] Gate 初始化消融完成（0.05 / 0.3 / 0.5）
- [x] 已输出 `docs/620/fog/ablation_audit/phase2_results.md`

## Phase 3: 扩展维度消融实验

- [x] 网络容量消融完成（64×4 / 64×6 / 128×4 / 128×6）
- [x] 已输出 `docs/620/fog/ablation_audit/phase3_capacity_results.md`
- [x] Loss 消融完成（SWD / NSWD / edge）
- [x] 已输出 `docs/620/fog/ablation_audit/phase3_loss_results.md`
- [x] DINO 与条件源消融完成（latent / target_dino_patches）
- [x] 已输出 `docs/620/fog/ablation_audit/phase3_dino_results.md`

## Phase 4: 综合历史 + 消融，产出设计取舍

- [x] 已汇总 Phase 2/3 所有消融结果
- [x] 已输出 `docs/620/fog/ablation_audit/results_summary.md`
- [x] 已对照 git 历史结论识别一致/冲突点
- [x] 已输出 `docs/620/fog/ablation_audit/history_vs_ablation.md`
- [x] 已对每个维度给出 KEEP / RESTORE / REMOVE / NEED_MORE_DATA
- [x] 已输出 `docs/620/fog/ablation_audit/design_decisions.md`

## Phase 5: 固化推荐配置并验证

- [x] 已生成 `configs/620_spatial_bridge_ablation_recommended.json`
- [x] 已输出 `docs/620/fog/ablation_audit/recommended_config.md`
- [x] 已训练推荐配置 1 epoch smoke 并跑通
- [x] 已跑 WFI 评估，确认 WFI < 0.40 且 CLIP-S ≥ 0.695
- [x] 已输出完整详尽报告 `docs/620/fog/ablation_audit/final_report.md`
- [x] 报告包含执行摘要、实验设计、并排对比表、历史对比、每个维度结论、推荐配置、未决问题
