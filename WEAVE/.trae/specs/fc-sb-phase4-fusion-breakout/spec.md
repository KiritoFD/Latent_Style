# FC-SB Phase 4 融合突破 Spec（v5 更新版）

## Why

6 个月 Pareto 死结的本质是 BASE LOCKING 结构性约束：clip↑/lpips↓/WFI↓ 三难困境。经过 FC-SB Phase 3 deepfix 修复三层 bug 链后，推理侧 U/V/T/DWT 机制已部分生效。v4 识别并修复了三个阻塞（config_override bug、endpoint_adain_scale guard、B2 V2 配置缺失），v5 在 v4 基础上：

1. **整合 Memory 全面回顾**（6/21-6/27 全部实验）：明确已探明机制清单与历史 Pareto 前沿点
2. **强化"先用已探明机制组合和调优"策略**：阶段 0 增加 U/V/scale 网格消融，训练前先用推理消融探明最佳参数组合
3. **重新规划 10 小时预算**：阶段 0 扩至 2h（消融探明）、阶段 1 缩至 3h、阶段 2 保持 3h、阶段 3 1.5h、阶段 4 0.5h

### 历史 Pareto 前沿点（Memory 全面回顾 6/21-6/27）

| 点 | clip_style | LPIPS | 来源 | 距双目标 (clip>0.74, lpips<0.35) |
|----|------------|-------|------|----------------------------------|
| E4-long ep5 | 0.727 | 0.581 | FC-SB spatial + RMSNorm + vmag2.0 (6/24) | clip 差 1.7%, lpips 差 66% |
| V3 (k16) | 0.7295 | 0.3963 | 推理侧 patch AdaIN (deepfix 后, 6/26) | clip 差 1.4%, lpips 差 13% |
| U4 (α0.1) | 0.7225 | 0.3660 | 推理侧 style 外推 (deepfix 后, 6/26) | clip 差 2.4%, lpips 差 4.6% |
| U1 (α0.2) | 0.7164 | 0.3735 | 推理侧 style 外推 (deepfix 后, 6/26) | clip 差 3.3%, lpips 差 6.7% |
| I7 baseline | 0.7031 | 0.3399 | endpoint FiLM=0.1 (6/26) | clip 差 5.2%, lpips 差 2.8% |
| 620_film_v5_hd512 | 0.7015 | 0.3382 | endpoint_film_hd512 (6/21) | clip 差 5.5%, lpips 差 3.4%, WFI 0.3906 |
| **B2 V2 ep1** | **0.6731** | **0.2781** | Spectral ODE 频域权重 (6/26) | clip 差 9.9%, **lpips 已达标** |
| B2 POC | 0.6676 | 0.2892 | Spectral ODE 基础 (6/26) | clip 差 11%, lpips 已达标 |
| E2 Two-Stage | ~0.611 | 0.3326 | 训练策略 S1style=16 (6/24) | clip 差 17%, lpips 已达标 |

### 已探明机制清单（Memory 回顾整合）

#### 推理侧机制（已验证生效，作用于 `integrate_transport`）

| 机制 | 参数 | 作用点 | 验证状态 | 最佳点 |
|------|------|--------|----------|--------|
| **U 方向** | `style_extrap_alpha` | 全局缩放 style_fiber `(1+α)` | ✅ deepfix 后生效 | U4 α0.1: clip=0.7225, lpips=0.3660 |
| **V 方向** | `patch_adain_kernel` | 局部 patch AdaIN (k=8/16/32) | ✅ deepfix 后生效 | V3 k16: clip=0.7295, lpips=0.3963 |
| **T 方向** | `multiband_adain_mode`, `mid/hh_adain_scale` | mid/hh 子带 AdaIN | ⚠️ 部分生效 (mid 影响 lpips, hh 影响 n1_ep_fiber_abs) | 待精调 |
| **DWT (D2)** | `lowpass_mode='dwt_haar'` | Haar 小波低频分离 | ✅ 代码已实现 | 待消融验证 |
| **tri_band_lock** | `tri_band_inference_lock` | 三频带锁定 | ⚠️ 待验证 | 待消融 |
| **endpoint_adain_scale** | 推理路径 AdaIN 强度 | endpoint AdaIN 缩放 | ✅ v4 修复 (guard 触发条件) | 待网格探测 |
| **CFG (R 方向)** | `cfg_scale` | classifier-free guidance | ❌ 单样式无效 | 5 样式待验证 |
| **W 方向** | `fiber_source_repulse_scale` | fiber source 排斥 | ⚠️ 有效但 margin 需 ≤ 10 | margin=20 损 LPIPS |

#### 训练侧机制（已实现，待训练验证）

| 机制 | 参数 | 作用点 | 实现状态 |
|------|------|--------|----------|
| **D2** | `lowpass_mode='dwt_haar'` | `forward._lowpass()` 训练-推理一致 | ✅ 已实现 |
| **D3** | `spectral_w_ll/lh/hl/hh` | per-subband FM loss 加权 | ✅ 已实现 |
| **D4** | `style_extrap_alpha` (训练版) | `forward` style_global `(1+α)` 缩放 | ✅ 已实现，训练路径不受 guard 阻塞 |
| **B2 频域权重** | `spectral_w_ll=0.3, spectral_w_hh=1.5` | Spectral ODE 核心旋钮 | ✅ V2 已验证 (6 个月首次双维度 Pareto 改进) |

#### 新机制（待探索，符合 FC-SB 理论）

| 机制 | 参数 | 理论依据 | 优先级 |
|------|------|----------|--------|
| **N1 多级 DWT** | `spectral_ode_levels=2/3` | B2 V2 频域权重成功，多级分解可能进一步解耦 | P1 |
| **N5 style_fiber 多级放大** | 多层级 `style_extrap_alpha` | 突破单点放大限制，强化 style 注入 | P1 |
| **N2 时频耦合** | `tf_schedule_enabled`, `tf_hh_max_scale` | 动态调整高频注入 | P2 |
| **N3 fiber repulsion** | `fiber_source_repulse_scale` | 增强 style 区分度 (W 教训：margin ≤ 10) | P2 |
| **N4 训练-推理联合** | 训练侧 N1 + 推理侧 patch_adain | 验证训练-推理一致性红利 | P2 |

### 双路突破策略（v5 沿用 v4，路径 B 更优先）

**路径 A（E4-long 路线，clip 优先）**：
- 起点：E4-long ep5 (0.727/0.581) + 推理侧 U4+V3 (0.7295/0.3963)
- 距 clip>0.74 仅差 1.4%，距 lpips<0.35 差 13%
- 策略：训练侧 D2/D3/D4 融合 + 推理侧 U/V 微调
- v4 阻塞已修复：U4 在 E4-long 上通过 `endpoint_adain_scale=1.0` 注入可生效

**路径 B（B2 V2 路线，lpips 已达标，更有希望）**：
- 起点：B2 V2 ep1 (0.6731/0.2781)，LPIPS 已远低于 0.35 目标
- 距 clip>0.74 差 9.9%，但 lpips 有 21% 余量可交换
- 策略：在 B2 V2 频域权重基础上，加训练侧 style_extrap_alpha (D4) + 推理侧 U/V
- v4 阻塞已解决：B2 V2 配置文件已确认存在

**路径 B 更有希望**：提升 clip 比降低 lpips 更可控（style_extrap_alpha、patch_adain、训练侧 D4 直接放大 style 信号），E4-long 的 lpips=0.581→0.35 需 -40% 难度极大。Memory 回顾进一步证实：B2 V2 是 6 个月首次双维度 Pareto 改进，频域权重是核心旋钮，早停 ep1 > 长训练 V4 ep12。

### v4 阻塞修复状态（v5 确认）

#### 阻塞 1：config_override bug — ✅ 已修复验证

- 第一层根因（已修复 ✅）：`_run_full_eval_for_checkpoint` 通过子进程调用 `run_evaluation.py`，`setattr` 修改的 config 对象不传递到子进程
- 第一次修复（已生效 ✅）：`src/run.py` 添加 `full_eval_config_override` 支持，`_p4_infer_ablation.py` 生成 override JSON
- 第二层根因（已识别 ✅）：`ModelConfig` dataclass 不含 `style_extrap_alpha`/`patch_adain_kernel`/`multiband_adain_mode`/`mid_adain_scale`/`hh_adain_scale`，进入 `extra` dict，通过 `_rehydrate_extra_attributes` setattr
- **传递链验证已通过** ✅

#### 阻塞 2：endpoint_adain_scale guard — ✅ 方案 A 已实施，待 D2 重跑验证

- 根因：`src/model620.py` L763 的 guard `if endpoint_adain_scale > 0.0` 嵌套了 `style_extrap_alpha` 应用（L785-786）
- E4-long config 默认 `endpoint_adain_scale=0.0`，导致 U 类消融被跳过
- **方案 A 已实施** ✅：`_p4_infer_ablation.py` L227-231 对含 `style_extrap_alpha > 0` 的消融组自动注入 `endpoint_adain_scale=1.0`
- **待验证**：重跑 D2 确认 Δclip > 0.001 vs D0 baseline 0.6799
- 备用方案 B：若方案 A 失败，修改 `model620.py` L763-786 将 style_extrap_alpha 移出 guard

#### 阻塞 3：B2 V2 配置文件缺失 — ✅ 已解决

- 现象：曾认为 `configs/620_spectral_v2_weights.json` 缺失
- 实际：文件已存在且参数正确（`spectral_w_ll=0.3, spectral_w_hh=1.5, num_epochs=8, batch_size=24`）
- Checkpoint 已确认：`exp/620_spectral_v2_weights/epoch_0001.pt`

### 目标

在 10 小时实验预算内突破 **clip_style > 0.74 且 LPIPS < 0.35** 双指标。**优先级：clip > 0.74 优于 LPIPS < 0.35**（用户 6/27 明确指示）。

策略（v5 调整）：
1. **阶段 0（2h）**：阻塞修复收尾 + **推理消融探明**（双 checkpoint × 单/联合/网格机制），先用推理消融找出最佳参数组合
2. **阶段 1（3h）**：基于阶段 0 探明的最佳推理参数，双路训练（路径 A T4 + 路径 B T5 优先 P0 并行）
3. **阶段 2（3h）**：新机制探索（N1 多级 DWT + N5 多级 style_fiber 优先 P1）
4. **阶段 3（1.5h）**：精调
5. **阶段 4（0.5h）**：失败兜底

## What Changes

### 阶段 0：阻塞修复收尾 + 推理消融探明（~2 小时，无需训练）

#### 0.0 阻塞修复收尾（v5 标记已完成状态）

**0.0.A：endpoint_adain_scale guard 修复** — ✅ 方案 A 已实施
- `_p4_infer_ablation.py` L227-231 对 U 类消融组自动注入 `endpoint_adain_scale=1.0`
- **待 D2 重跑验证** Δclip > 0.001
- 失败兜底：方案 B（修改 model620.py 移出 guard）

**0.0.B：B2 V2 配置文件恢复** — ✅ 已解决
- `configs/620_spectral_v2_weights.json` 已存在且参数正确

**0.0.C：移除 [P4_DEBUG] 临时调试输出** — ✅ 已完成
- `model620.py` L623-638 调试代码已移除，typo 修复保留

#### 0.1 推理消融矩阵（双 checkpoint × 单/联合/网格机制）— **v5 强化"消融探明"**

**Checkpoint 1: E4-long ep5（D0 baseline = 0.6799/0.6283）**

单机制消融（6 组）：
- D1: + DWT (`lowpass_mode='dwt_haar'`)
- D2: + U4 (`style_extrap_alpha=0.1` + `endpoint_adain_scale=1.0`) — **v4 修复后重跑**
- D3: + V3 (`patch_adain_kernel=16`)
- D4: + V6 (`patch_adain_kernel=32`)
- D5: + T (`multiband_adain_mode='two_level'`, `mid_adain_scale=0.3`, `hh_adain_scale=0.3`)
- D6: + tri_band_lock (`tri_band_inference_lock=true`)

联合机制消融（4 组）：
- D7: U4 + V3 联合（含 endpoint_adain_scale=1.0）
- D8: U4 + V3 + DWT 三联合
- D9: U4 + V3 + DWT + T 四联合
- D9b: U4 + V3 + DWT + T + `endpoint_adain_scale=0.5`（探测 adain_scale 强度）

**Checkpoint 2: B2 V2 ep1（D0-B2 baseline，待复现，预期 ~0.6731/0.2781）**
- D10: B2 V2 baseline 复现
- D11: B2 V2 + U4 (`style_extrap_alpha=0.1` + `endpoint_adain_scale=1.0`) — **关键：验证 clip 能否突破 0.74**
- D12: B2 V2 + V3 (`patch_adain_kernel=16`)
- D13: B2 V2 + U4 + V3 联合
- D14: B2 V2 + U4 + V3 + DWT 三联合
- D15: B2 V2 + U4 (α=0.2 更激进) + V3 + DWT — 探测能否直接破 0.74

**v5 新增：网格消融探明（基于阶段 0 初步结果，选最佳联合组合做网格）**
- D16: U 方向 α 网格（0.05/0.1/0.15/0.2/0.3）× 最佳联合组合 — 5 组
- D17: V 方向 k 网格（8/16/32/48）× 最佳联合组合 — 4 组
- D18: endpoint_adain_scale 网格（0.5/1.0/1.5）× 最佳联合组合 — 3 组

**产出**：双 checkpoint 消融对比表 + 网格探明最佳推理参数组合，重点验证 B2 V2 + U/V 能否推 clip 突破 0.74。

### 阶段 1：训练侧融合（~3 小时）— 代码改动已完成，训练待启动

解决"训练/推理路径不对称"问题：U/V/T/DWT 目前只在推理路径 `integrate_transport` 生效，`forward` 训练路径已有对应代码（D2/D3/D4 已实现）。

**代码改动（已完成 ✅）**：
- **D2 改动** ✅：`src/model620.py` `forward._lowpass()` 读取 `lowpass_mode`，支持 `dwt_haar`/`wavelet`/`avg_pool`
- **D3 改动** ✅：`src/losses620.py` `SpatialBridgeObjective620.forward` 新增 per-subband FM loss 分支
- **D4 改动** ✅：`forward` 训练路径加 style_extrap_alpha 训练版，训练-推理完全一致
- **D4-fix** ✅：训练路径 style_extrap_alpha 应用（L443-445）独立于 endpoint_adain_scale guard，不受阻塞

**训练实验（双路线并行，基于阶段 0 探明的最佳推理参数）**：

路径 A（E4-long 路线，基于 `exp/p3_remote_10h/e4_long_10ep/config.json`）：
- T1: E4-long + `endpoint_style_hidden_dim=512`（架构容量基线）
- T2: T1 + 训练侧 DWT lowpass（D2，训练-推理一致）
- T3: T1 + 频域 FM loss（D3，w_ll=0.3 锁低频保 lpips, w_hh=1.5 放高频提 clip）
- T4: T1 + D2 + D3 + 训练侧 style_extrap_alpha=0.1 + `endpoint_adain_scale=1.0`（D4，全融合，**P0 最高优先级**）

路径 B（B2 V2 路线，基于 `configs/620_spectral_v2_weights.json`）— v5 确认配置已恢复
- T5: B2 V2 + 训练侧 D2 + D4 + `endpoint_adain_scale=1.0`（**P0 最高优先级**）
- T6: B2 V2 + D2 + D3（w_ll=0.5 强锁低频, w_hh=2.0 强放高频）+ D4
- T7: B2 V2 + D4 + `endpoint_adain_scale=1.0`（仅 D4，最小改动验证 D4 在频域架构上的效果）

**优先级**：T4 和 T5 并列 P0（双路突破），T6/T7 次之，T1/T2/T3 用于路径 A 消融验证。

### 阶段 2：新机制探索（~3 小时）

基于阶段 1 双路最佳点，探索符合 FC-SB 理论的新机制：
- **N1: 多级 DWT** (`spectral_ode_levels=2/3`) 在 `lp()` 函数实现 — 强化频域解耦，**P1 提升优先级**（B2 V2 已证明频域权重是核心旋钮）
- **N5: style_fiber 多级放大** — 在 forward 多个 fiber 层级应用 style_extrap_alpha，突破单点放大限制，**P1 提升优先级**
- **N2: 时频耦合调度** (`tf_schedule_enabled=true`, `tf_hh_max_scale=1.5`) — 动态调整高频注入
- **N3: fiber source repulsion** (`fiber_source_repulse_scale=0.1/0.3`) — 增强 style 区分度（注意 W2b 教训：margin ≤ 10，过大损 LPIPS）
- **N4: 训练侧 N1 + 推理侧 patch_adain 联合** — 验证训练-推理一致性红利

### 阶段 3：精调（~1.5 小时）

在阶段 1/2 双路最佳点附近做参数微调，目标逼近 clip>0.74, lpips<0.35：
- U 方向 α 微调 (0.05/0.1/0.15/0.2/0.3)
- V 方向 k 微调 (8/16/32/48)
- endpoint_adain_scale 网格 (0.5/1.0/1.5)
- w_ll/w_hh 频域权重网格 (0.1/0.3/0.5 × 1.0/1.5/2.0)
- 早停点选择（epoch 1/3/5/8）

### 阶段 4：失败兜底（~0.5 小时，按需）

若上述未达双指标，记录 Pareto 前沿推进，更新 project_memory，给出下一阶段方向（如 mixture-of-experts / per-style adapter / 更激进的频域解耦 / 跨 checkpoint ensemble / 训练侧 endpoint_adain_scale 改造）。

## Impact

- **Affected specs**:
  - `fc-sb-phase3-deepfix`（前置，U/V/T/W 修复完成，推理侧机制已生效）
  - `fc-sb-tuning-deepdive`（前置，I7 baseline）
  - `anti-degen-hardcore`（前置，E4-long 基线）
  - `fc-sb-breakthrough`（前置，B2 Spectral ODE V2 基线）
- **Affected code**:
  - `src/model620.py`：`forward._lowpass()` 扩展（D2 ✅），`forward` 加 style_extrap_alpha 训练版（D4 ✅，训练路径不受 guard 阻塞 ✅）；方案 B 待命（修改 L763 guard）
  - `src/losses620.py`：`SpatialBridgeObjective620.forward` 加 per-subband FM loss（D3 ✅）
  - `src/run.py`：`_run_full_eval_for_checkpoint` 支持 `config_override`（✅ 已验证生效）
  - `src/utils/inference.py`：`LGTInference` config_override_path 合并逻辑（✅ 已验证）
  - `_p4_infer_ablation.py`：override JSON 生成对 U 类消融自动加 `endpoint_adain_scale: 1.0`（✅ L227-231 已实施）
  - `configs/`：T1-T4 已生成（✅，T4 已补 endpoint_adain_scale ✅），T5-T7 已生成（✅），`configs/620_spectral_v2_weights.json` 已确认存在（✅）；N1-N5 待生成
  - `exp/p4_fusion_breakout/`：新实验目录
- **Affected infrastructure**:
  - 远程 I 盘 `I:\Github\Latent_Style\SchrodingerBridge`
  - Windows native Python（避开 WSL 不稳定）
  - 12GB VRAM 约束（batch=16 安全，batch=24 需探测）

## ADDED Requirements

### Requirement: endpoint_adain_scale guard 修复验证（v5 修正状态）

系统 SHALL 确保 `style_extrap_alpha` 推理参数在 `endpoint_adain_scale` 未在 checkpoint config 中显式设置时仍能生效，通过 override 同时设置 `endpoint_adain_scale > 0.0`。**v5 状态**：方案 A 已实施，待 D2 重跑验证。

#### Scenario: 推理参数生效验证
- **WHEN** 通过 `config_override` 设置 `style_extrap_alpha=0.1` 且 `endpoint_adain_scale=1.0` 并评估
- **THEN** `model620.py` `integrate_transport` L763 guard 触发，style_extrap_alpha 在 L785-786 应用
- **AND** 评估结果 clip_style 与 D0 baseline (0.6799) 有可测量差异（Δ > 0.001）

### Requirement: B2 V2 配置恢复（v5 已完成）

系统 SHALL 通过从 I 盘远程同步或基于 620_spectral_poc.json 重建，恢复 `configs/620_spectral_v2_weights.json`，使其能加载 `exp/620_spectral_v2_weights/epoch_0001.pt` checkpoint。**v5 状态**：✅ 已确认文件存在且参数正确。

#### Scenario: V2 配置加载验证
- **WHEN** 用恢复后的 V2 配置通过 `LGTInference` 加载 epoch_0001.pt
- **THEN** 无 key mismatch 错误，model_cfg.spectral_w_ll=0.3, spectral_w_hh=1.5

### Requirement: 推理侧机制系统化消融探明（v5 强化）

系统 SHALL 在 E4-long ep5 和 B2 V2 ep1 两个 checkpoint 上对已探明推理侧机制做单机制 + 联合机制 + 网格消融，产出对比表，**训练前先用推理消融探明最佳参数组合**。

#### Scenario: E4-long 单机制消融
- **WHEN** 对 E4-long ep5 checkpoint 应用单个推理侧机制（如 `lowpass_mode='dwt_haar'`）
- **THEN** 评估 clip_style 和 LPIPS，与 D0 baseline (0.6799/0.6283) 对比，记录增益方向

#### Scenario: B2 V2 单机制消融
- **WHEN** 对 B2 V2 ep1 checkpoint 应用 U4 (`style_extrap_alpha=0.1` + `endpoint_adain_scale=1.0`)
- **THEN** 评估 clip_style 和 LPIPS，与 D0-B2 baseline (~0.6731/0.2781) 对比
- **AND** 验证 B2 V2 + U4 能否推 clip 突破 0.74（关键路径 B 验证点）

#### Scenario: 联合机制消融
- **WHEN** 同时应用 U4+V3+DWT+T 多机制
- **THEN** 评估并记录联合增益是否大于单机制之和（验证作用点正交性）

#### Scenario: 网格消融探明（v5 新增）
- **WHEN** 阶段 0 初步结果确定最佳联合组合后
- **THEN** 对 U 方向 α (0.05/0.1/0.15/0.2/0.3)、V 方向 k (8/16/32/48)、endpoint_adain_scale (0.5/1.0/1.5) 做网格消融
- **AND** 产出最佳推理参数组合，作为阶段 1 训练配置基础

### Requirement: 训练侧 DWT lowpass 支持

系统 SHALL 在 `SpatialBridge620.forward` 训练路径中支持 `lowpass_mode='dwt_haar'`，与推理路径 `integrate_transport.lp()` 使用相同的正交 Haar DWT 实现。

#### Scenario: 训练时 DWT 一致性
- **WHEN** 配置 `lowpass_mode='dwt_haar'` 并训练
- **THEN** `forward._lowpass()` 调用 `spectral620.dwt2_haar/idwt2_haar` 做 fiber/base 分离，训练-推理使用同一低频分离算法

### Requirement: 频域 per-subband FM loss

系统 SHALL 在 `SpatialBridgeObjective620.forward` 中支持 per-subband FM loss，按 `spectral_w_ll/w_lh/w_hl/w_hh` 加权。

#### Scenario: 频域权重生效
- **WHEN** 配置 `spectral_w_ll=0.3, spectral_w_hh=1.5` 并训练
- **THEN** FM loss 被分解为 4 个子带的加权 L2 loss，低频权重小（锁低频保 LPIPS），高频权重大（放高频提 clip）

### Requirement: 训练-推理 style_extrap_alpha 一致性（v5 已验证）

系统 SHALL 在 `forward` 训练路径中对 style_global 应用 `(1+α)` 缩放，与推理路径 `integrate_transport` 行为一致。**v5 状态**：✅ 训练路径 L443-445 独立于 endpoint_adain_scale guard，不受阻塞。

#### Scenario: 训练时 style 外推
- **WHEN** 配置 `style_extrap_alpha=0.1` + `endpoint_adain_scale=1.0` 并训练
- **THEN** 训练路径 style_global 被放大 1.1 倍，模型学习到外推后的 style 分布，推理时无需额外调整

### Requirement: B2 V2 路线训练实验

系统 SHALL 在 B2 V2 Spectral ODE 基础上添加训练侧 D2/D3/D4 融合，验证能否在保持 LPIPS<0.35 的前提下推 clip>0.74。

#### Scenario: B2 V2 + D4 训练
- **WHEN** 在 B2 V2 配置上添加 `style_extrap_alpha=0.1` + `endpoint_adain_scale=1.0` 并训练
- **THEN** 评估 clip_style 是否突破 0.74，LPIPS 是否保持在 0.35 以内
- **AND** 若达成双指标，标记为路径 B 突破点

### Requirement: 10 小时实验预算控制

系统 SHALL 在 10 小时实验预算内完成 5 个阶段的所有实验，单次训练+评估 ≤ 30 分钟。

#### Scenario: 预算超限保护
- **WHEN** 单次训练耗时 > 30 分钟或总耗时接近 10 小时
- **THEN** 优先完成当前阶段已启动实验，跳过剩余低优先级实验，记录原因

### Requirement: Pareto 前沿推进报告

系统 SHALL 在每个阶段结束时输出当前 Pareto 前沿点，与历史最佳对比。

#### Scenario: 双指标达成
- **WHEN** 任一实验配置评估结果 clip_style > 0.74 且 LPIPS < 0.35
- **THEN** 标记为目标达成，记录配置细节，进入精调阶段

#### Scenario: 双指标未达但 Pareto 推进
- **WHEN** 任一实验配置 clip_style > 0.74 但 LPIPS ≥ 0.35，或 LPIPS < 0.35 但 clip_style ≤ 0.74
- **THEN** 记录为 Pareto 前沿推进点，继续下一实验

## MODIFIED Requirements

### Requirement: SpatialBridge620 forward 低频分离
原 `forward._lowpass()` 仅支持 avg_pool。修改后 SHALL 读取 `lowpass_mode` 配置，支持 `avg_pool | wavelet | dwt_haar` 三种模式，与推理路径 `integrate_transport.lp()` 行为一致。（D2 改动已完成 ✅）

### Requirement: SpatialBridgeObjective620 FM loss
原 FM loss 为全频带 L2。修改后 SHALL 支持 per-subband 分解，按 `spectral_w_ll/lh/hl/hh` 加权。（D3 改动已完成 ✅）

### Requirement: SpatialBridge620 forward style 注入（v5 已验证）
原 `forward` 训练路径不应用 style_extrap_alpha。修改后 SHALL 在 style_global 提取后应用 `(1+α)` 缩放，与推理路径一致。**v5 状态**：D4 改动已完成 ✅，训练路径 L443-445 独立于 endpoint_adain_scale guard，不受阻塞 ✅。

## REMOVED Requirements

无。本 spec 全部为新增/修改，不删除已有功能。
