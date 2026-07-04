# Tasks: 620 全面消融实验与设计审计

## Phase 0: Git 历史调研（已完成）

- [x] Task 0.1: 列出并检查所有相关分支
  - [x] 当前分支 `codex/620-spatial-bridge`
  - [x] `codex/tokenizer-clean-c3058eab`
  - [x] `origin/attn`, `origin/SWD`, `origin/Style8_Moment+SWD`, `origin/re-SWD`, `origin/multistep-texture`, `origin/exp/style-injection-priority-proto-sep` 等
- [x] Task 0.2: 提取各分支 commit message 中的实验结果
- [x] Task 0.3: 对比关键分支的代码差异
- [x] Task 0.4: 读取 `docs/620/` 中的历史结果
- [x] Task 0.5: 输出 `docs/620/fog/ablation_audit/git_history_digest.md`

## Phase 1: 基线整理与消融矩阵设计（已完成）

- [x] Task 1.1: 整理历史基线信息
  - [x] Round 1 `base_swd8`
  - [x] 白化修复前 `620_film_v5_gated_local_smoke`
  - [x] 此前自评最优 `endpoint_film_hd512`
  - [x] 输出 `docs/620/fog/ablation_audit/baseline_table.md`
- [x] Task 1.2: 设计消融维度与实验矩阵
  - [x] 核心维度：attention、style_film、endpoint、gate_init
  - [x] 扩展维度：capacity、loss、DINO、condition_source
  - [x] 输出 `docs/620/fog/ablation_audit/ablation_matrix.md`
- [x] Task 1.3: 准备统一评估脚本
  - [x] 创建 `tools/generate_ablation_configs.py`
  - [x] 创建 `tools/run_ablation_batch.py`
  - [x] 确认 `tools/run_eval_with_wfi.py` 可在本地稳定运行

## Phase 2: 核心维度消融实验（已完成）

- [x] Task 2.1: Attention 机制消融
  - [x] gated / softmax / gated_raw / relu2 / style_select / sparsemax
  - [x] 输出 `docs/620/fog/ablation_audit/phase2_results.md`
- [x] Task 2.2: StyleFiLM 消融
  - [x] true / false
- [x] Task 2.3: Endpoint 结构消融
  - [x] velocity / endpoint_lowhigh × film on/off × hd128/256/512
- [x] Task 2.4: Gate 初始化消融
  - [x] 0.05 / 0.3 / 0.5

## Phase 3: 扩展维度消融实验（部分完成）

- [x] Task 3.1: 网络容量消融
  - [x] 64×4 / 64×6 / 128×4 / 128×6
  - [x] 输出 `docs/620/fog/ablation_audit/phase3_capacity_results.md`
- [x] Task 3.2: Loss 消融
  - [x] `single_step_swd_weight`: 0 / 2 / 8 / 16
  - [x] `swd_noise_sigma`: 0 / 0.02
  - [x] `single_step_edge_weight`: 0 / 0.1
  - [x] 输出 `docs/620/fog/ablation_audit/phase3_loss_results.md`
- [x] Task 3.3: DINO 与条件源消融
  - [x] `style_condition_source`: latent / target_dino_patches
  - [x] 如果支持，测试 DINO adapter / intrinsic cross-attention 的复测
  - [x] 输出 `docs/620/fog/ablation_audit/phase3_dino_results.md`

## Phase 4: 综合历史 + 消融，产出设计取舍

- [x] Task 4.1: 汇总所有消融结果
  - [x] 读取 Phase 2/3 所有 `wfi_eval_report.json`
  - [x] 计算每个维度的独立贡献和交互影响
  - [x] 生成汇总表格与可视化
  - [x] 输出 `docs/620/fog/ablation_audit/results_summary.md`
- [x] Task 4.2: 综合 git 历史结论与当前消融
  - [x] 对照 git_history_digest.md 中的历史教训
  - [x] 识别历史结论与当前结果一致 / 冲突的地方
  - [x] 输出 `docs/620/fog/ablation_audit/history_vs_ablation.md`
- [x] Task 4.3: 产出设计取舍报告
  - [x] 对每个维度给出 KEEP / RESTORE / REMOVE / NEED_MORE_DATA
  - [x] 输出 `docs/620/fog/ablation_audit/design_decisions.md`

## Phase 5: 固化推荐配置并验证

- [x] Task 5.1: 生成最小有效配置
  - [x] 基于 design_decisions.md 生成 `configs/620_spatial_bridge_ablation_recommended.json`
  - [x] 移除所有 REMOVE 模块
  - [x] 恢复所有 RESTORE 模块的历史默认值
  - [x] 输出 `docs/620/fog/ablation_audit/recommended_config.md`
- [x] Task 5.2: 验证推荐配置
  - [x] 训练 1 epoch smoke
  - [x] 跑 WFI 评估
  - [x] 确认 WFI < 0.40 且 CLIP-S ≥ 0.695
- [x] Task 5.3: 产出完整详尽报告
  - [x] 整合基线、消融矩阵、历史对比、结果、取舍、推荐配置
  - [x] 输出 `docs/620/fog/ablation_audit/final_report.md`

# Task Dependencies

```
Phase 0 ──→ Phase 1 ──→ Phase 2 ──┐
                                  ├──→ Phase 4 ──→ Phase 5
Phase 1 ──→ Phase 3 ──────────────┘
```

- Phase 2 与 Phase 3 可并行
- Phase 4 必须等待 Phase 2/3 完成
- Phase 5 必须等待 Phase 4 完成
