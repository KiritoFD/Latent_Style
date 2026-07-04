# Tasks

## Phase 1: 初步减法消融（已完成）

- [x] Task 1: 确认 T5 config 实际配置键名并生成砍除清单
  - [x] 读取 T5 config (`exp/p4_fusion_breakout/t5_b2v2_d2_d4/config.json`)
  - [x] 输出 `configs/ablations/629_subtractive/prune_manifest.json`
- [x] Task 2-3: Stage 1 砍 13 死 loss 训练验证 → clip=0.7293, lpips=0.3451
- [x] Task 5-6: Stage 2 砍 2 有害 loss 训练验证 → clip=0.7292, lpips=0.3441
- [x] Task 7-8: Stage 3a 砍 9 arch (D4-D12) → clip=0.7299, **lpips=0.3894（LPIPS 灾难）**
- [x] Task 9-10: Stage 3b 砍 8 arch (D13-D21) 累积 → clip=0.7301, lpips=0.3895
- [x] Task 11-12: Stage 3c 砍 6 arch (D22-D30) 累积 → clip=0.7300, lpips=0.3895
- [x] Task 14-15: Stage 4 生成 clean_base_v2.json (23 arch cuts) + 最终训练 → clip=0.7299, lpips=0.3895
- [x] Task 17: Phase 8F 正交消融完成

**Phase 1 结论**：S3a 引入 LPIPS 灾难（+0.049），当前 clean_base_v2.json 不可接受。需 Phase 2 诊断。

## Phase 2: 诊断 + 修正（已完成）

- [x] Task P2.1: 更新 spec.md 反映 LPIPS 双指标 + S3a 排除
- [x] Task P2.2: 写诊断 runner（Test B + Test C + Test E）
- [x] Task P2.3: 启动诊断 runner 远程执行（schtasks 解耦）
- [x] Task P2.4: 等待 + 监控诊断 runner 完成（3 实验 × ~7min = ~21min）
- [x] Task P2.5: 分析诊断结果，决定最终砍除集合
  - [x] Test C: clip=0.7285 FAIL
  - [x] Test B: clip=0.7299, lpips=0.3420 **PASS** → 候选配置
  - [x] Test E: clip=0.7285 FAIL
- [x] Task P2.6: S3a per-item rollback（已完成）
  - [x] D4_lowpass: clip=0.7300, lpips=0.3896 **FAIL**（LPIPS 灾难罪魁）
  - [x] D5_skip_clean: clip=0.7299, lpips=0.3420 PASS → 累积
  - [x] D6_skip_blur: clip=0.7299, lpips=0.3420 PASS → 累积
  - [x] D7_decoder_hp: clip=0.7298, lpips=0.3419 PASS → 累积
  - [x] D8_residual_gain: clip=0.7298, lpips=0.3420 PASS → 累积
  - [x] D9_no_residual: clip=0.7297, lpips=0.3420 PASS → 累积
  - [x] D10_style_gate: clip=0.7298, lpips=0.3420 PASS → 累积
  - [x] D11_affine_gamma: clip=0.7298, lpips=0.3420 PASS → 累积
  - [x] D12_affine_beta: clip=0.7298, lpips=0.3421 PASS → 累积
  - [x] 结论：D4 是 S3a LPIPS 灾难唯一罪魁，D5-D12 共 8 项 SAFE

## Phase 3: 最终交付（已完成）

- [x] Task 14v2: 生成最终 clean_base_v2.json（22 cuts = 14 Test B + 8 S3a safe）
  - [x] 永久排除 D4（model.lowpass_mode）
  - [x] 已远程保存到 `configs/clean_base_v2.json`
- [x] Task 14v2.1: 同步 clean_base_v2.json 到本地仓库
- [x] Task 15v2: 22 cuts 整体验证训练
  - [x] 22 cuts 整体训练 10 epoch → clip=0.7298, lpips=0.3421
  - [x] 验证 clip ≥ 0.7293 且 lpips ≤ 0.3453 → **PASS** ✓
  - [x] 确认无组合负面交互
- [x] Task 16: 创建全面说明文档
  - [x] 编写 `docs/CLEAN_BASE_V2.md`（22 cuts 最终版本）
  - [x] 包含：Phase 1 实测数据 + S3a 排除原因（D4 罪魁）+ Phase 2 诊断结果 + S3a per-item rollback + 22 cuts 验证 + 最终砍除清单（22 项）+ Pareto 对比 + 使用指南
  - [x] 同步到远程
- [x] Task 17: 更新 tasks.md / checklist.md 最终勾选
- [x] Task 18: 更新 spec.md（S3a 排除修正：仅 D4 + 22 cuts 砍除清单 + 22 cuts 验证结果）

# Task Dependencies

- Task P2.2 依赖 Task P2.1 ✓
- Task P2.3 依赖 Task P2.2 ✓
- Task P2.4 依赖 Task P2.3 ✓
- Task P2.5 依赖 Task P2.4 ✓
- Task P2.6 依赖 Task P2.5 ✓
- Task 14v2 依赖 Task P2.6 ✓
- Task 15v2 依赖 Task 14v2 ✓
- Task 16 依赖 Task 15v2 ✓
- Task 17 依赖 Task 16 ✓
- Task 18 依赖 Task 17 ✓

# 最终状态

**全部任务已完成。** clean_base_v2.json（22 cuts）已验证 PASS，文档已归档。
