# Checklist

## Phase 1: 初步减法消融（已完成）

- [x] 砍除清单 JSON 已生成（prune_manifest.json）
- [x] Stage 1 训练完成（S1_dead_loss: clip=0.7293, lpips=0.3451）
- [x] Stage 2 训练完成（S2_harmful_loss: clip=0.7292, lpips=0.3441）
- [x] Stage 3a 训练完成（S3a_arch_batch1: clip=0.7299, **lpips=0.3894** ← 灾难）
- [x] Stage 3b 训练完成（S3b_arch_batch2: clip=0.7301, lpips=0.3895）
- [x] Stage 3c 训练完成（S3c_arch_batch3: clip=0.7300, lpips=0.3895）
- [x] Phase 1 最终 clean_base_v2.json 生成（23 arch cuts, LPIPS=0.3895, **拒绝**）

## Phase 2: 诊断 + 修正（已完成）

- [x] spec.md 已更新 v2 判定逻辑（D0_control ep10 噪声基准 + LPIPS 双指标）
- [x] spec.md 已添加 S3a 排除 requirement（修正：仅排除 D4）
- [x] spec.md 已添加 Phase 2 诊断测试矩阵 requirement
- [x] spec.md 已添加 Phase 2 实测数据 + S3a per-item rollback 结果 + 22 cuts 验证
- [x] tasks.md 已更新 Phase 2 任务列表（全部完成）
- [x] 诊断 runner 已写（_629_diagnostic_runner.py）
- [x] Test C 配置已生成（baseline + S1 + S2 + S3b + S3c, 29 cuts）
- [x] Test B 配置已生成（baseline + S3b + S3c, 14 cuts）
- [x] Test E 配置已生成（baseline + S1 + S2, 15 cuts）
- [x] 诊断 runner 已远程启动（schtasks）
- [x] Test C 训练 + 评估完成 → clip=0.7285, lpips=0.3415 → FAIL
- [x] Test B 训练 + 评估完成 → clip=0.7299, lpips=0.3420 → **PASS** ✓
- [x] Test E 训练 + 评估完成 → clip=0.7285, lpips=0.3415 → FAIL
- [x] 最终砍除集合已决策：Test B（14 arch cuts）+ S3a safe items（8 项）= 22 cuts

## Phase 2 S3a Per-Item Rollback（已完成）

- [x] D4_lowpass: clip=0.7300, lpips=0.3896 → **FAIL**（LPIPS 灾难罪魁）
- [x] D5_skip_clean: clip=0.7299, lpips=0.3420 → PASS（累积）
- [x] D6_skip_blur: clip=0.7299, lpips=0.3420 → PASS（累积）
- [x] D7_decoder_hp: clip=0.7298, lpips=0.3419 → PASS（累积）
- [x] D8_residual_gain: clip=0.7298, lpips=0.3420 → PASS（累积）
- [x] D9_no_residual: clip=0.7297, lpips=0.3420 → PASS（累积）
- [x] D10_style_gate: clip=0.7298, lpips=0.3420 → PASS（累积）
- [x] D11_affine_gamma: clip=0.7298, lpips=0.3420 → PASS（累积）
- [x] D12_affine_beta: clip=0.7298, lpips=0.3421 → PASS（累积）
- [x] LPIPS 罪魁已定位：**D4（model.lowpass_mode）是唯一罪魁**
- [x] 可安全加入的单项 arch cuts 已识别：D5-D12 共 8 项

## Phase 3: 最终交付（已完成）

- [x] configs/clean_base_v2.json 已生成（远程 + 本地，22 cuts 最终版本）
- [x] configs/clean_base_v2.json 已同步到本地仓库
- [x] 22 cuts 整体验证训练完成（10 epoch, clip=0.7298, lpips=0.3421）
- [x] 最终评估 clip ≥ 0.7293 → 0.7298 PASS
- [x] 最终评估 lpips ≤ 0.3453 → 0.3421 PASS
- [x] 最终配置中**不包含** D4（model.lowpass_mode 保留 baseline "dwt_haar"）
- [x] 最终配置**包含** D5-D12 共 8 项 S3a safe items（已验证 PASS）
- [x] docs/CLEAN_BASE_V2.md 已创建，包含：
  - [x] Phase 1 实测数据表（含 S3a LPIPS 灾难记录）
  - [x] S3a 排除原因（D4 是罪魁，非 9 项组合交互）
  - [x] Phase 2 诊断结果（Test B/C/E）
  - [x] S3a per-item rollback 完整结果（9 项）
  - [x] 22 cuts 整体验证结果
  - [x] 最终砍除清单（22 项 = 14 Test B + 8 S3a safe）
  - [x] Pareto 对比表（clean_base_v2 vs T5 baseline vs Phase 1 拒绝版 vs Test B）
  - [x] 使用指南
- [x] 文档已同步到远程 I 盘
- [x] tasks.md / checklist.md 最终勾选完成

## 核心模块保留验证

- [x] model.contract_family = '620_spectral_ode'（保留）
- [x] model.endpoint_adain_scale > 0（保留）
- [x] model.style_extrap_alpha > 0（保留）
- [x] bridge.spectral_w_ll > 0（保留，唯一有效谱 loss）

## 砍除完整性验证

- [x] 验证 14 项 arch cuts（S3b+S3c）开关全改为 prune_to 值
- [x] 验证 8 项 S3a safe items（D5-D12）开关全改为 prune_to 值
- [x] D4（model.lowpass_mode）保留 baseline 值 "dwt_haar"（不砍除）
- [x] 死 loss 权重保留 baseline 值（不砍除，因 S1+S2 组合 FAIL）
- [x] harmful loss（spectral_lh/hl）权重保留 baseline 1.0（不砍除，因 S1+S2 组合 FAIL）
- [x] 配置中无残留的 LPIPS 有害项（D4 已排除）
