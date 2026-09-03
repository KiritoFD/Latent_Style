# SchrodingerBridge (LBM) Documentation

## Guides

| Document | Description |
|----------|-------------|
| [Quick Start](quickstart.md) | Training, evaluation, config usage |
| [Architecture](architecture.md) | Code structure, loss components, data flow |
| [Remote Server](remote_server.md) | SSH, schtasks, GameViewer, SaMST pipeline |

## Mathematical Foundations

| Document | Description |
|----------|-------------|
| [Maths Overview](maths/README.md) | 理论推导、数学模型和实验证据索引 |
| [Code-Faithful Model](maths/MODEL.md) | 状态空间、端点映射、损失函数形式化 |
| [Theory Reset](maths/THEORY_RESET_2026-05-16.md) | 围绕最小可验证声明的理论重建 |
| [Decision Tree](maths/DECISION_TREE_AND_EXPERIMENT_PLAN.md) | 数学决策树和实验计划 |
| [Reflections (ZH)](maths/REFLECTIONS.md) | 实验反思与理论修正 |

## Development Reports

| Document | Description |
|----------|-------------|
| [Phase 1 Cleanup Report](cleanup_report.md) | What was removed, why, and what remains |
| [A/B Test Log](ablation_log.md) | Phase 2 experiments and results |
| [Known Issues](known_issues.md) | clip_style ceiling, inference config, eval failures |

## Experiment Reports (Historical)

| Report | Date | Topic |
|--------|------|-------|
| [Model/Modules/Losses Technical Note](presentations/2026-05-21-model-modules-losses-and-results.md) | 2026-05-21 | 当前模型架构、已确认模块、Loss 与实验结果汇总说明 |
| [Scale Experiment](scale_experiment_report.md) | 2026-05 | 1024px WikiArt 27-genre evaluation |
| [256 Diffeomorphic Tangent Progress](experiments/2026-05-20-256-diffeomorphic-tangent-progress.md) | 2026-05-20 | 256px diffeomorphic stroke / tangent warp 实验进展与结论 |
| [Baseline Reproduction](experiments/2026-05-11-baseline-reproduction-lab-notes.md) | 2026-05-11 | Baseline SOTA reproduction |
| [Theory Reset Plan](experiments/2026-05-16-theory-reset-and-fast-k05-plan.md) | 2026-05-16 | OT-Coupled Latent Flow Matching theory |
| [Phase 1 Diagnostic](experiments/2026-05-16-phase1-diagnostic-launcher.md) | 2026-05-16 | Diagnostic probe framework |
| [Reproducibility Report (ZH)](repro_report_zh/) | 2026-05 | Chinese reproducibility documentation |
