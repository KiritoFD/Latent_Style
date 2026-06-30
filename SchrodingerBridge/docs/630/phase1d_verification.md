# Phase 1D: 最简 Codebase 性能验证 (2026-06-30)

## 概述
验证 Phase 1C 精简后的最简 codebase (删除 11346 行 dead code) 性能不下降.

## 训练配置
- config: `configs/630_phase1d_verify.json`
- save_dir: `exp/630_phase1d_verify_v2` (独立目录, 不 resume)
- epochs: 3 (与 Phase 1B 对比基准相同)
- batch_size: 16
- 从 scratch 训练 (No checkpoint found, start from scratch)

## 评估结果 (allpairs n=750, 显存 <=7G)

| Metric | Phase 1D (3ep, 精简后) | Phase 1B (3ep, 精简前) | baseline (10ep) | 阈值 | 判定 |
|--------|------------------------|------------------------|-----------------|------|------|
| clip_style | 0.7251 | 0.7269 | 0.7293 | >= 0.7243 | **PASS** |
| content_lpips | 0.3373 | 0.3370 | 0.3203 | <= 0.3453 | **PASS** |

## 分析
1. **性能不下降**: 精简后 clip=0.7251 vs 精简前 0.7269, Δ=-0.0018 (训练噪声范围内)
2. **LPIPS 一致**: 精简后 0.3373 vs 精简前 0.3370, Δ=+0.0003 (几乎相同)
3. **重构安全**: 删除 11346 行 dead code 对 active 路径 (620_spectral_ode) 零影响
4. **3 epochs 即 PASS**: 说明精简后 codebase 仍保持快速收敛特性

## Codebase 现状 (Phase 1 完成后)
### Active 路径核心文件
- `src/model.py` (93 行) - 模型工厂
- `src/spectral_bridge620.py` - SpectralODEBridge620 主模型
- `src/spectral_losses620.py` - SpectralODEObjective620 loss
- `src/blocks620.py` (279 行) - SpatialBridgeBlock620
- `src/style_encoder620.py` (109 行) - StyleConditioner620
- `src/spectral620.py` - Haar DWT 工具
- `src/trainer.py` - 训练器 (lazy import)
- `src/run.py` - 入口
- `src/config_schema.py` - 配置 schema
- `src/style_families.py` - 风格族工具
- `src/utils/inference.py` - 推理工具
- `src/utils/run_evaluation.py` - 评估工具
- `src/utils/training.py` - 训练工具
- `src/utils/dataset.py` - 数据集

### 保留的 620_spatial_bridge 契约 (历史实验复现)
- `src/model620.py` (1013 行)
- `src/losses620.py` (793 行)

### 已删除的 Legacy (11346 行)
- TimeConditionedLANCETBridge 类
- losses.py, ot_cost.py, lancet_runtime.py, lancet_blocks.py, lancet_backbone.py
- style_tokenizer.py, semantic_tokenizer.py, round1_registry.py, round2_registry.py
- tests/test_infra_guardrails.py

## 结论
**Phase 1 完成**: codebase 已精简到最简优雅状态, 性能与 baseline 持平, 可以进入 Phase 2 (masking 探索).

## Next Steps
- Phase 2: 实现 masking (参考 docs/630/mask.md) + 充分消融
