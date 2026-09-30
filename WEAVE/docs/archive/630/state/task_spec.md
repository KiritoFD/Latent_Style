# Task Spec: 630 Long-Horizon Cleanup + Masking + Exploration

## Goal
在已清理的 clean_base_v2 codebase 基础上,继续通过减法消融识别并删除无效模块/Loss,确保模型达到绝对最简最佳性能;之后实现 Masking(参考 docs/630/mask.md),并基于理论探索其他提升方法。每阶段必须训练验证性能不下降,文档记录到 docs/630/,git 提交。

## Baseline Reference (4070 Laptop, 2026-06-30)
- Config: configs/clean_base_v2_local.json
- Model: SpectralODEBridge620 (~903K params)
- clip_style = 0.7293 (PASS >= 0.7243)
- content_lpips = 0.3203 (PASS <= 0.3453)
- Smoke: GPU 33.9 MB, loss ~= 4.59

## Hard Constraints (project_memory)
- 所有消融配置统一 batch_size=24 (但本机 baseline 是 16,以本地 baseline 为准)
- 训练 Patience=2, max=10, 至少 5 epochs
- 显存控制在 9-11G
- 数据集路径 I 盘 (本地: G:/GitHub/Latent_Style/Dataset/distinct5_512)
- DataLoader: num_workers=0, pin_memory=False
- 命令添加 30s timeout
- 无效代码/机制确认后直接删除(不 ablate)
- 优化用条件编译,避免影响其他测试
- 不允许远程 GPU,本地重训

## Milestones
- M1: 深度审计完成,识别所有候选 dead code (state/findings.jsonl)
- M2: Phase 1 减法消融完成,每项删除通过 smoke + 短训练验证性能不下降
- M3: Codebase 达到"绝对最简最佳性能 + 软件工程规范"标准
- M4: Phase 2 Masking 实现 + 充分消融
- M5: Phase 3 探索其他提升方法
- M6: 所有阶段文档化 + git 提交完成

## Success Criteria
1. 最终 codebase 行数显著减少,无 dead code,无未使用配置项
2. clip_style >= 0.7243, content_lpips <= 0.3453 始终满足
3. Masking 实现完成且经过充分消融验证
4. 每阶段有文档记录和 git commit

## Direction Diversity Rule
新方向必须与已尝试方向不同。stale_count>=2 时切换结构约束,非战术参数。

## Round Cap
单次工作 session 上限 15 rounds 或 30 分钟。
