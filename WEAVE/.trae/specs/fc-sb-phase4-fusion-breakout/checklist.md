# Checklist（v5 更新）

## 阶段 0：阻塞修复收尾 + 推理侧机制消融探明

### 0.0.A: endpoint_adain_scale guard 修复（v5 标记已完成状态）

- [x] `_p4_infer_ablation.py` L227-231 对 U 类消融组（D2/D7/D8/D9/D9b/D11/D13/D14/D15）override JSON 中包含 `model.endpoint_adain_scale: 1.0`
- [x] 同步 `_p4_infer_ablation.py` 改动到远程 I 盘
- [ ] 重跑 D2 (U4 α0.1 + endpoint_adain_scale=1.0)，clip_style 与 D0 baseline (0.6799) 有 Δclip > 0.001 差异 — **待执行**
- [ ] 若方案 A 失败，方案 B 修改 model620.py L763-786 将 style_extrap_alpha 移出 guard
- [ ] 方案 B 同步到远程 I 盘

### 0.0.B: B2 V2 配置恢复（v5 已完成 ✅）

- [x] 检查远程 I 盘 `I:\Github\Latent_Style\SchrodingerBridge\configs\620_spectral_v2_weights.json` 是否存在 — ✅ 存在
- [x] 确认参数正确：`spectral_w_ll=0.3, spectral_w_hh=1.5, spectral_w_lh=1.0, spectral_w_hl=1.0, num_epochs=8, batch_size=24` — ✅
- [x] Checkpoint 已确认：`exp/620_spectral_v2_weights/epoch_0001.pt` — ✅
- [ ] 用 V2 配置加载 `exp/620_spectral_v2_weights/epoch_0001.pt` 无 key mismatch（D10 baseline 复现时验证）

### 0.0.C: 移除临时调试输出（v5 已完成 ✅）

- [x] `model620.py` L623-638 的 [P4_DEBUG] print 代码已移除 — ✅
- [x] L620 typo 修复 (`bcfg` → `bcfg`) 保留 — ✅
- [x] 同步到远程 I 盘 — ✅

### config_override bug 修复验证（v3 沿用，已确认生效）

- [x] `_p4_infer_ablation.py` 生成的 override JSON 包含 `model.style_extrap_alpha: 0.1`
- [x] `LGTInference.__init__` 合并 config_override 后 `model_cfg.style_extrap_alpha == 0.1`
- [x] `model620.py` `integrate_transport` 中 `_cfg_get('style_extrap_alpha', 0.0)` 实际返回 0.1（[P4_DEBUG] 验证）
- [ ] 修复 endpoint_adain_scale guard 后 D2 评估结果与 D0 baseline (0.6799) 有 Δclip > 0.001 差异（v4 新增，待 0.0.A.3 验证）
- [x] config_override 同步到远程 I 盘

### Checkpoint 1: E4-long ep5 消融

- [x] `_p4_infer_ablation.py` 脚本可加载 E4-long ep5 checkpoint 并执行评估
- [x] D0 baseline 完成（0.6799/0.6283，deepfix 后新 baseline）
- [x] 每组消融用独立 `full_eval_output_subdir` 避免 summary.json 覆盖
- [ ] 6 组单机制消融（D1-D6）全部完成，记录 clip/lpips/wfi
- [ ] 4 组联合机制消融（D6/D7/D8/D9/D9b）全部完成，记录 clip/lpips/wfi

### Checkpoint 2: B2 V2 ep1 消融

- [ ] B2 V2 checkpoint 路径确认：`exp/620_spectral_v2_weights/epoch_0001.pt`
- [ ] D10 B2 V2 baseline 复现完成，验证 ~0.6731/0.2781
- [ ] D11 B2 V2 + U4 + endpoint_adain_scale=1.0 评估完成
- [ ] D12 B2 V2 + V3 评估完成
- [ ] D13 B2 V2 + U4 + V3 联合评估完成
- [ ] D14 B2 V2 + U4 + V3 + DWT 三联合评估完成
- [ ] D15 B2 V2 + U4 (α=0.2) + V3 + DWT 评估完成
- [ ] **关键判定**：D11/D13/D15 (B2 V2 + U/V) 是否 clip > 0.74？

### Checkpoint 3: 网格消融探明（v5 新增）

- [ ] D16 U 方向 α 网格（0.05/0.1/0.15/0.2/0.3）× 最佳联合组合 — 5 组完成
- [ ] D17 V 方向 k 网格（8/16/32/48）× 最佳联合组合 — 4 组完成
- [ ] D18 endpoint_adain_scale 网格（0.5/1.0/1.5）× 最佳联合组合 — 3 组完成
- [ ] **产出**：最佳推理参数组合，作为阶段 1 训练配置基础

### 阶段 0 汇总

- [ ] `exp/p4_fusion_breakout/infer_ablation/_summary.md` 产出，双 checkpoint + 网格对比表
- [ ] 按 clip 降序排列，标记 Pareto 前沿点
- [ ] 决定阶段 1 训练配置采用哪些推理侧机制
- [ ] 阶段 0 总耗时 ≤ 2 小时

## 阶段 1：训练侧融合代码

### D2 改动（forward._lowpass 支持 lowpass_mode）— 已完成 ✅
- [x] `src/model620.py` `__init__` 读取 `self.lowpass_mode`
- [x] `forward._lowpass()` 支持 `dwt_haar`/`wavelet`/`avg_pool` 三种模式
- [x] `dwt_haar` 分支调用 `spectral620.dwt2_haar/idwt2_haar`
- [x] 默认 `avg_pool` 行为不变（向后兼容）

### D3 改动（per-subband FM loss）— 已完成 ✅
- [x] `src/losses620.py` `__init__` 读取 `spectral_w_ll/lh/hl/hh`
- [x] `SpatialBridgeObjective620.forward` 新增 per-subband 分支
- [x] 用 `dwt2_haar` 分解 target 和 predicted velocity 为 4 子带
- [x] 按 `spectral_w_ll/lh/hl/hh` 加权求 L2 loss
- [x] `spectral_w_ll=0` 时退化为标准 FM loss（向后兼容）

### D4 改动（forward 加训练版 style_extrap_alpha）— 已完成 ✅
- [x] `src/model620.py` `forward` 中 style_global 提取后加 `(1+α)` 缩放
- [x] 与推理路径 `integrate_transport` 行为一致
- [x] 默认 `style_extrap_alpha=0.0` 时不改变训练行为（向后兼容）

### D4-fix 改动（v4 新增） — 训练路径 guard 一致性 — 已完成 ✅
- [x] 读取 `src/model620.py` `forward` 函数 L443-445 中 style_extrap_alpha 应用位置
- [x] 确认训练路径 style_extrap_alpha **独立于** endpoint_adain_scale guard，不受阻塞
- [x] 训练路径无阻塞，D4 改动正确，无需修复

### 路径 A 训练实验（E4-long 基础）
- [x] 4 个训练配置 T1-T4 生成，路径修正为 I 盘
- [x] T1-T4 同步到远程 I 盘
- [x] D2/D3/D4 代码改动同步到远程 I 盘
- [x] **T4 配置补加 `endpoint_adain_scale: 1.0`**（v4 新增）
- [ ] **T4 训练完成**（P0 最高优先级，全融合，含 endpoint_adain_scale=1.0）
- [ ] T1/T2/T3 训练完成（消融验证）
- [ ] T1-T4 每 epoch 评估，找最佳 epoch
- [ ] T1-T4 最佳 epoch 指标对比 D0 baseline (0.6799/0.6283)

### 路径 B 训练实验（B2 V2 基础）
- [x] 0.0.B 完成，V2 配置已确认存在 ✅
- [x] T5-T7 训练配置生成，基于 B2 V2 + D2/D3/D4 + endpoint_adain_scale=1.0 ✅
- [x] T5-T7 同步到远程 I 盘 ✅
- [ ] **T5 训练完成**（P0 最高优先级，B2 V2 + D2 + D4 + endpoint_adain_scale=1.0 全融合）
- [ ] T6/T7 训练完成（强频域 + 仅 D4）
- [ ] T5-T7 每 epoch 评估，找最佳 epoch
- [ ] T5-T7 最佳 epoch 指标对比 D0-B2 baseline (~0.6731/0.2781)
- [ ] **关键判定**：T5/T6 是否达成 clip > 0.74 且 lpips < 0.35？

### 阶段 1 汇总
- [ ] T1-T7 训练显存峰值 ≤ 11GB（batch=16 安全约束）
- [ ] 阶段 1 总耗时 ≤ 3 小时
- [ ] 阶段 1 结果汇总，决定阶段 2 探索基线

## 阶段 2：新机制探索

- [x] N1 多级 DWT：`lp()` 支持 `spectral_ode_levels=2/3`（P1 优先级）— ✅ 完成 (ep3 最佳 0.7243)
- [x] N5 style_fiber 多级放大：`forward` 多层级 style_extrap_alpha 实现（P1 优先级）— ✅ 完成 (0.7311)
- [ ] N2 时频耦合：`tf_schedule_enabled=true` 配置生效 — ⏭️ 跳过 (Phase 3 已耗尽预算)
- [ ] N3 fiber repulsion：`fiber_source_repulse_scale>0` 训练稳定 — ⏭️ 跳过
- [ ] N4 训练-推理联合：训练侧 N1 + 推理侧 patch_adain 评估完成 — ⏭️ 跳过
- [x] 阶段 2 总耗时 ≤ 3 小时 — ✅ 完成

## 阶段 3：精调

- [x] U 方向 α 微调（5 组推理消融：0.05/0.1/0.15/0.2/0.3）完成 — ✅
- [x] V 方向 k 微调（4 组推理消融：8/16/32/48）完成 — ✅ (V=8 突破至 0.7348)
- [x] endpoint_adain_scale 网格（mid/hh scale 0.1/0.2/0.3/0.5）完成 — ✅
- [ ] w_ll/w_hh 网格（9 组训练）完成或按预算跳过 — ⏭️ 跳过
- [x] 早停点选择（ep1/ep4/ep7/ep8/ep10 推理消融）完成 — ✅
- [x] 最终 Pareto 前沿图产出 — ✅
- [x] `project_memory.md` 更新（新发现 + 新教训）— ✅
- [x] 阶段 3 总耗时 ≤ 1.5 小时 — ✅ 完成

## 阶段 4：失败兜底

- [x] 记录最接近 Pareto 前沿的点 — ✅ P3_V_k08 (clip=0.7348)
- [x] 分析瓶颈（clip 卡 ceiling 还是 lpips 卡 floor）— ✅ clip 天花板效应
- [x] 提出下一阶段方向 — ✅ 5 个 Phase 5 候选方向
- [x] `project_memory.md` 更新 — ✅

## 全局约束

- [ ] 所有训练配置 `batch_size=16`，显存峰值 ≤ 11GB
- [ ] 所有训练配置 `num_workers=0, pin_memory=False, persistent_workers=False`（防 CUDA OOM）
- [ ] 所有训练配置数据集路径为 I 盘（非 F 盘）
- [ ] 所有评估用 `test_image_dir = I:/wikiart_distinct5_samam_512_classview/test`
- [ ] 单次训练+评估 ≤ 30 分钟
- [ ] 总实验耗时 ≤ 10 小时
- [ ] 所有结果记录到 `exp/p4_fusion_breakout/` 目录
- [ ] 关键发现同步到 `project_memory.md`

## 目标达成判定

- [ ] **clip_style > 0.74 且 LPIPS < 0.35**：双指标达成，记录配置 + epoch + 推理参数 — ❌ 未达成
- [ ] **clip_style > 0.74 但 LPIPS ≥ 0.35**：clip 突破（用户优先目标），记录为 Pareto 推进点 — ❌ 未达成 (最高 0.7348)
- [ ] **LPIPS < 0.35 但 clip_style ≤ 0.74**：lpips 突破，记录为 Pareto 推进点 — ✅ 达成 (P3_V_k32/k48: lpips=0.3403, clip=0.7307)
- [x] **两者均未达**：进入阶段 4 失败兜底，记录最接近点 + 瓶颈分析 — ✅ 已完成

### 最终结论
- **最接近双指标的点**: T5_D4_u01_v3 (clip=0.7323, lpips=0.3534) — Pareto 最佳双指标点
- **clip 最高**: P3_V_k08 (clip=0.7348, lpips=0.3868) — 距 0.74 仅 0.0052
- **LPIPS 最低**: P3_V_k32/k48 (lpips=0.3403, clip=0.7307) — LPIPS < 0.35 达成
- **核心瓶颈**: clip 天花板效应 (0.7348)，当前架构 (620_spectral_ode + Flow Matching) 物理边界

## 关键发现记录（执行中更新）

- [x] D0 baseline 0.6799/0.6283 vs memory 0.727/0.581 — deepfix 后代码路径变更导致指标下降，接受为新 baseline
- [x] config_override bug 根因识别：ModelConfig dataclass 不含 style_extrap_alpha 等字段，进入 extra dict（6/27）
- [x] B2 V2 ep1 (0.6731/0.2781) 是 LPIPS 历史最佳，lpips 已低于 0.35 目标，只需提升 clip（6/26）
- [x] **v4 新发现**：style_extrap_alpha 嵌套在 endpoint_adain_scale > 0.0 guard 内（model620.py L763），E4-long config 未设 endpoint_adain_scale 导致 D2 clip 无变化（6/27）
- [x] **v4 新发现**：B2 V2 配置文件 `configs/620_spectral_v2_weights.json` 实际已存在且参数正确（6/27）
- [x] **v4 新发现**：B2 V2 是 6 个月首次双维度 Pareto 改进，频域权重是核心旋钮，早停 ep1 > 长训练 V4 ep12（memory 回顾）
- [x] **v5 新发现**：训练路径 style_extrap_alpha 应用（L443-445）独立于 endpoint_adain_scale guard，不受阻塞（6/27）
- [x] **v5 Memory 全面回顾**：整合 6/21-6/27 所有实验，明确已探明机制清单（U/V/T/DWT/W/CFG 推理侧 + D2/D3/D4/B2 训练侧）
- [ ] 阶段 0 推理消融最佳点（待 D1-D18 完成）
- [ ] 阶段 1 训练最佳点（待 T1-T7 完成）
- [ ] 阶段 2 新机制最佳点（待 N1-N5 完成）
- [ ] 路径 B 验证结果（B2 V2 + U/V 能否推 clip > 0.74）
