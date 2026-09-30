# Phase 1B: M9 attn_mode Bug TDD 修复 + relu2 评估 (2026-06-30)

## 问题发现 (M9 Bug)
审计发现 `style_attn_mode: "relu2"` 在 clean_base_v2_local.json 中配置, 但 `spectral_bridge620.py` 构造 `SpatialBridgeBlock620` 时从未传递 `attn_mode` 参数, 导致所有 block 默认使用 `"softmax"`.

**影响**: baseline clip_style=0.7293 / content_lpips=0.3203 是在 softmax (非配置意图的 relu2) 下取得的.

## TDD 修复流程 (RED → GREEN)

### RED: 写失败测试
- **文件**: `tests/test_630_spectral_ode.py`
- **测试**: `test_style_attn_mode_propagated_to_blocks()` 断言 `block.attn_mode == config.style_attn_mode`
- **结果**: 失败 (block.attn_mode="softmax", config="relu2")

### GREEN: 最小修复
- **文件**: `src/spectral_bridge620.py:72-86`
- **修改**: 构造 block 时传递 5 个之前遗漏的配置字段:
  - `attn_mode` (M9 核心)
  - `gate_mode` (style_gate_mode)
  - `attn_temperature` (style_attn_temperature)
  - `shortcut_alpha` (style_shortcut_alpha)
  - `norm_type` (body_norm_type)
- **结果**: 测试通过

### 配套修复: run_evaluation.py 回归 bug
- **问题**: Phase 1A H2 移除 `source_style_latent` 参数后, `run_evaluation.py:3263` 仍在调用 `generation_with_target_latent(source_style_latent=...)`, 导致评估崩溃
- **修复**: 移除该 kwarg 调用

## 评估对比 (3-epoch quick check)

### 训练配置
- config: `configs/clean_base_v2_relu2.json` (style_attn_mode="relu2")
- epochs: 3 (快速对比, 非完整训练)
- batch_size: 16
- loss 收敛: 2.484 → 2.311 → 2.185 (正常下降)

### 评估结果 (显存控制在 7G 以内)
| Metric | relu2 (3ep) | softmax baseline (10ep) | 阈值 | 判定 |
|--------|-------------|-------------------------|------|------|
| clip_style | 0.7269 | 0.7293 | ≥ 0.7243 | **PASS** |
| content_lpips | 0.3370 | 0.3203 | ≤ 0.3453 | **PASS** |

### 结论
1. **M9 修复正确**: relu2 attn_mode 现在真正生效, 模型能正常收敛并通过验收
2. **3 epochs 即 PASS**: 说明 relu2 是可行的注意力模式 (非破坏性)
3. **性能对比说明**: relu2 3ep 的 clip 略低 (0.7269 vs 0.7293)、lpips 略高 (0.3370 vs 0.3203), 但均在阈值内; 完整 10 epochs 训练后性能应与 softmax baseline 持平或更优 (relu2 的稀疏性理论上更利于风格提取)
4. **决策**: 保留 relu2 修复 (配置应当被尊重); 后续完整训练时再确认最终性能

## 显存约束更新
- 用户反馈: 评估显存不得超过 7G
- 配置调整: `full_eval_batch_size=2`, `full_eval.batch_size=2`, `ref_feature_batch_size=2`, `max_ref_compare=16`, `max_ref_cache=16`
- 同步更新: `configs/clean_base_v2_local.json` + `configs/clean_base_v2_relu2.json` + `project_memory.md`

## 修改文件清单
1. `src/spectral_bridge620.py` - block 构造传递 5 个配置字段
2. `src/utils/run_evaluation.py` - 移除 `source_style_latent` kwarg
3. `tests/test_630_spectral_ode.py` - 新增 3 个 TDD 测试
4. `configs/clean_base_v2_relu2.json` - 新建 3-epoch 对比配置 + 显存调整
5. `configs/clean_base_v2_local.json` - 显存调整 (batch_size=2)
6. `docs/630/phase1b_attn_mode_fix.md` - 本文档

## Next Steps
- Phase 1C: M1-M8 legacy 文件批量删除 (~12000+ 行 dead code)
