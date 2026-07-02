# Phase 4A-1: Dead Code Removal (2026-07-01)

## 背景

Phase 3 完成后, codebase 已精简到 14 个 active src 文件。本阶段通过深度审计 `spectral_losses620.py` 的 placeholder metrics, 识别并删除确认的 dead code。

## 审计方法

使用 search subagent 审计 `trainer.py` 和 `run.py` 对 `SpectralODEObjective620.compute()` 返回的所有 metric keys 的引用情况。判定标准:
- **可删除**: trainer/run 均无引用, 且 `trainer.py` 的 `_avg()` guard + `setdefault` 兜底保证删除后行为不变
- **保留**: trainer/run 有引用 (LIVE logging 指标)
- **条件性**: 取决于运行时配置

## 删除清单

| 对象 | 类型 | 删除理由 |
|------|------|---------|
| `spectral_brownian_noise_scale` | placeholder metric | trainer/run 无引用 (Brownian 分支已删除) |
| `loss_type` metric key | 0/1 tensor | trainer/run 无引用 (config 层 loss_type 字符串保留) |
| `loss_fm` alias | loss.detach() 别名 | trainer/run 无引用 |
| `loss_fm_total` | loss.detach() 别名 | trainer/run 无引用 |
| `compute_debug` 方法 | dead method | 仅归档脚本调用, 训练主路径 dead code |
| `loss_fn.last_debug` 赋值 | 内部状态 | trainer 读 model.last_debug, 不读 loss_fn.last_debug |

## 保留清单

| 对象 | 保留理由 |
|------|---------|
| `flow` | LIVE: trainer.py tqdm postfix + run.py epoch 日志 |
| `terminal_swd/ot_cost/kinetic_energy/curvature` | 兼容 legacy 日志格式 (删除后显示 0.0 不变, 但保留为可选) |
| `loss_fm_spectral_ll/lh/hl` | 调试价值, 监控 per-subband loss |
| `t_mean` | 调试价值, 监控时间采样 |
| `style_dino_active` | 监控 DINO patches 是否可用 |
| `update_weights_for_epoch` | trainer 调用 (返回值仅日志, 但方法不能删) |

## 验证

### Smoke Test
- **配置**: `configs/630_phase3_mask_random_50_10ep.json --smoke-only`
- **结果**: ALL PASS
  - Model params: 903,248 (与 baseline 一致)
  - loss=4.556082 (baseline 4.59, 噪声范围)
  - Backward OK, Optimizer step OK
  - GPU allocated: 33.1 MB, reserved: 222.3 MB

## 修改文件

1. `src/spectral_losses620.py` - 删除 6 项 dead code, 添加注释说明保留理由
2. `docs/630/phase4a1_dead_code_removal.md` - 本文档
3. `docs/630/state/progress.json` - 更新到 Phase 4
4. `docs/630/state/directions_tried.json` - 记录方向

## 结论

Stage 4A-1 完成。删除 6 项确认的 dead code, smoke test 验证 forward/backward 正常。性能与 baseline 一致 (loss 在噪声范围内)。

## Next Steps

- Stage 4A-2: 验证未验证的超参数 (spectral_w_ll, attn_temperature, shortcut_alpha)
- Stage 4B: 频率掩码方案 C (分频 tokenizer)
