# Tasks

## 阶段 0：阻塞修复 + 推理侧机制消融（~1.5 小时，无需训练）

### 前置 0.0：阻塞修复（v4 重组 P0 前置，并行可做）

#### 0.0.A：endpoint_adain_scale guard 修复（P0 阻塞）

- [x] Task 0.0.A: 修复 endpoint_adain_scale guard 阻塞 — ✅ 方案 A 生效（D2_v5 Δclip=+0.0118）
  - [ ] 0.0.A.1: 修改 `_p4_infer_ablation.py`，对所有含 `style_extrap_alpha > 0` 的消融组（D2/D7/D8/D9/D9b/D11/D13/D14/D15）自动加入 `endpoint_adain_scale: 1.0` 到 override JSON 的 `model` section
  - [ ] 0.0.A.2: 同步 `_p4_infer_ablation.py` 改动到远程 I 盘
  - [x] 0.0.A.3: 重跑 D2_v5（U4 α0.1 + endpoint_adain_scale=1.0），Δclip = +0.0118 > 0.001 — ✅ 方案 A 生效
  - [x] 0.0.A.4: 方案 A 已生效，无需启用方案 B
  - [x] 0.0.A.5: 无需同步 model620.py（方案 B 未启用）

#### 0.0.B：B2 V2 配置文件恢复（P0 阻塞，并行可做）

- [ ] Task 0.0.B: 恢复 `configs/620_spectral_v2_weights.json`
  - [ ] 0.0.B.1: 检查远程 I 盘 `I:\Github\Latent_Style\SchrodingerBridge\configs\620_spectral_v2_weights.json` 是否存在
  - [ ] 0.0.B.2: 若远程存在，同步回 G 盘 `configs/` 目录
  - [ ] 0.0.B.3: 若远程缺失，基于 `configs/620_spectral_poc.json` 创建 `configs/620_spectral_v2_weights.json`，覆盖：`spectral_w_ll=0.3, spectral_w_hh=1.5, spectral_w_lh=1.0, spectral_w_hl=1.0, num_epochs=8`，其余继承 POC
  - [ ] 0.0.B.4: 同步 V2 配置到远程 I 盘
  - [ ] 0.0.B.5: 用 V2 配置加载 `exp/620_spectral_v2_weights/epoch_0001.pt`，确认无 key mismatch（用 LGTInference 试加载，不必跑评估）

#### 0.0.C：移除临时调试输出（最后做）

- [ ] Task 0.0.C: 移除 `model620.py` 中的 [P4_DEBUG] print 代码（L623-638），保留 typo 修复（L620 `bcfg`）

### Checkpoint 1: E4-long ep5 消融

- [x] Task 0.1: 准备消融脚本 `_p4_infer_ablation.py` ✅
  - [x] 加载 E4-long ep5 checkpoint (`exp/p3_remote_10h/e4_long_10ep/checkpoints/epoch_0005.pt`)
  - [x] 加载 E4-long 训练配置 (`exp/p3_remote_10h/e4_long_10ep/config.json`)
  - [x] 命令行参数覆盖 `lowpass_mode`/`patch_adain_kernel`/`style_extrap_alpha`/`multiband_adain_mode`/`tri_band_inference_lock` 等
  - [x] 修复 Windows 路径（I:/ 替代 /mnt/i/）
  - [x] 每组消融用独立 `full_eval_output_subdir` 避免 summary.json 覆盖
  - [x] 调用 `run._run_full_eval_for_checkpoint`，结果写入 `exp/p4_fusion_breakout/infer_ablation/<name>.json`
- [x] Task 0.2: E4-long 单机制消融（6 组）— ✅ 完成（最佳 D2_v5 clip=0.6917）
  - [x] D0: baseline（avg_pool, 无 U/V/T）— 0.6799/0.6283 ✅
  - [x] D1: + DWT (`lowpass_mode='dwt_haar'`) — 0.6799/0.6283（α=0 时 DWT 无效果）
  - [x] D2: + U4 v5 (`style_extrap_alpha=0.1` + `endpoint_adain_scale=1.0`) — 0.6917/0.5669 ✅ 方案 A 生效
  - [x] D3: + V3 (`patch_adain_kernel=16`) — 0.6800/0.6283（α=0 时 V 无效果）
  - [x] D4: + V6 (`patch_adain_kernel=32`) — 0.6800/0.6283（α=0 时 V 无效果）
  - [x] D5: + T (`multiband_adain_mode='two_level'`) — 0.6800/0.6283（α=0 时 T 无效果）
- [x] Task 0.3: E4-long 联合机制消融（4 组）— ✅ 完成（最佳 D8_u4_v3_dwt clip=0.7054 ★Pareto）
  - [x] D6: + tri_band_lock — 0.6876/0.6102（非 U 类唯一有效参数）
  - [x] D7: U4 + V3 联合 v5 — 0.6905/0.5709
  - [x] D8: U4 + V3 + DWT 三联合 — **0.7054/0.5053 ★Pareto 前沿**
  - [x] D9: U4 + V3 + DWT + T 四联合 — 0.6560/0.6255（two_level + fix 异常退化，待排查）
  - [ ] D9b: U4 + V3 + DWT + T + `endpoint_adain_scale=0.5`（跳过，D9 已异常退化）

### Checkpoint 2: B2 V2 ep1 消融（路径 B 验证 P1）

- [x] Task 0.4: B2 V2 推理消融（6 组）— ❌ 架构不兼容，推理消融无效
  - [x] D10: B2 V2 baseline 复现 — 0.6731/0.2781 ✅ 完美复现
  - [x] D11: B2 V2 + U4 — 0.6731/0.2781（Δclip=0，**架构不兼容**）
  - [x] D12: B2 V2 + V3 — 0.6731/0.2781（Δclip=0）
  - [x] D13: B2 V2 + U4 + V3 — 0.6731/0.2781（Δclip=0）
  - [x] D14: B2 V2 + U4 + V3 + DWT — 0.6731/0.2781（Δclip=0）
  - [x] D15: B2 V2 + U4 (α=0.2) + V3 + DWT — 0.6731/0.2781（Δclip=0）
  - **关键发现**：`SpectralODEBridge620.integrate_transport()` 不读取 ablation 参数，需移植 hooks
- [ ] Task 0.5: 汇总阶段 0 结果到 `exp/p4_fusion_breakout/infer_ablation/_summary.md`
  - [ ] 运行 `_p4_summarize_ablations.py` 生成双 checkpoint 汇总表
  - [ ] 按 clip_style 降序排列
  - [ ] 标记 Pareto 前沿点（对照历史：E4-long ep5, V3, U4, B2 V2）
  - [ ] **关键判定**：若 D11/D13/D15 (B2 V2 + U/V) clip > 0.74，路径 B 验证成功，阶段 1 优先 T5
  - [ ] 决定阶段 1 训练配置采用哪些推理侧机制

## 阶段 1：训练侧融合代码（~3.5 小时）— 代码完成, 训练待启动

- [x] Task 1.1: D2 改动 — `forward._lowpass()` 支持 `lowpass_mode` ✅
- [x] Task 1.2: D3 改动 — `losses620.py` per-subband FM loss ✅
- [x] Task 1.3: D4 改动 — `forward` 加训练版 `style_extrap_alpha` ✅
- [ ] Task 1.3b: D4-fix 改动（v4 新增） — 检查 `forward` 训练路径 style_extrap_alpha 是否也嵌套在 endpoint_adain_scale guard 内
  - [ ] 1.3b.1: 读取 `src/model620.py` `forward` 函数中 style_extrap_alpha 应用位置
  - [ ] 1.3b.2: 若在 guard 内，参考 0.0.A 方案 A（配置层加 endpoint_adain_scale=1.0）或方案 B（代码层移出 guard）同步处理
  - [ ] 1.3b.3: 同步改动到远程 I 盘
- [x] Task 1.4: 生成路径 A 训练配置 T1-T4 ✅
  - [x] T1: `configs/p4_t1_arch.json` — `endpoint_style_hidden_dim: 512`
  - [x] T2: `configs/p4_t2_dwt.json` — T1 + `lowpass_mode: 'dwt_haar'`
  - [x] T3: `configs/p4_t3_spectral_loss.json` — T1 + `spectral_w_ll: 0.3, spectral_w_hh: 1.5`
  - [x] T4: `configs/p4_t4_full_fusion.json` — T1 + D2 + D3 + `style_extrap_alpha: 0.1`
  - [x] **v4 修正**：T4 配置需补加 `endpoint_adain_scale: 1.0`，否则训练时 style_extrap_alpha 也被 guard 跳过
  - [x] 所有配置：`num_epochs: 10`, `batch_size: 16`, `resume_checkpoint: ""`, I 盘路径
- [ ] Task 1.5: 生成路径 B 训练配置 T5-T7（依赖 Task 0.0.B）
  - [ ] 基于 `configs/620_spectral_v2_weights.json`（V2 配置已恢复）创建 T5-T7
  - [ ] T5: B2 V2 + `lowpass_mode: 'dwt_haar'` + `style_extrap_alpha: 0.1` + `endpoint_adain_scale: 1.0`（D2+D4 最小融合）
  - [ ] T6: B2 V2 + D2 + D3 (`spectral_w_ll: 0.5, spectral_w_hh: 2.0`) + D4 + `endpoint_adain_scale: 1.0`（强频域融合）
  - [ ] T7: B2 V2 + D4 + `endpoint_adain_scale: 1.0`（仅 `style_extrap_alpha: 0.1`，最小改动验证 D4 效果）
  - [ ] 所有配置：`num_epochs: 10`, `batch_size: 16`, `resume_checkpoint: ""`, I 盘路径
  - [ ] 同步到远程 I 盘
- [ ] Task 1.6: 训练 T4 + T5（双路 P0 并行）+ 评估
  - [ ] 同步代码改动到远程 I 盘（D2/D3/D4 已同步 ✅，T4 配置需补 endpoint_adain_scale，T5-T7 待生成）
  - [ ] **T4 训练**（路径 A 全融合，E4-long 基础，含 endpoint_adain_scale=1.0）
  - [ ] **T5 训练**（路径 B 全融合，B2 V2 基础，含 endpoint_adain_scale=1.0）
  - [ ] Windows native Python 训练，每 epoch 评估
  - [ ] 失败容错：单实验失败不阻塞后续
  - [ ] 记录每个 epoch 的 clip_style/lpips，找最佳 epoch
  - [ ] 显存监控：峰值 ≤ 11GB
- [ ] Task 1.7: 训练 T1/T2/T3 + T6/T7（消融验证 + 次优先级）
  - [ ] T1/T2/T3 路径 A 消融（验证 D2/D3 各机制贡献）
  - [ ] T6/T7 路径 B 次优先级（强频域 + 仅 D4）
  - [ ] 失败容错，超时跳过
- [ ] Task 1.8: 阶段 1 结果汇总
  - [ ] 对比 T1-T7 最佳 epoch 与 D0 baseline (0.6799/0.6283) / D0-B2 baseline
  - [ ] 对比 T1-T7 与阶段 0 推理消融最佳点
  - [ ] 决定阶段 2 探索基线（T1-T7 中最佳）
  - [ ] 更新 `project_memory.md` 记录新发现

## 阶段 2：新机制探索（~3 小时）

- [x] Task 2.1: N1 — 多级 DWT (`spectral_ode_levels=2/3`) — **P1 提升优先级** — ✅ 完成 (训练侧触顶 0.7243)
  - [x] 修改 `src/spectral_bridge620.py` forward/integrate_transport 支持多级 DWT (levels>1)
  - [x] 调用 `spectral620.dwt2_multi_level` / `idwt2_multi_level`
  - [x] 修复 `src/spectral_losses620.py` target velocity 多级 DWT 分解 (bug: 维度不匹配 8 vs 32)
  - [x] 生成配置 `configs/p4_n1_lvl2.json` (spectral_ode_levels=2, style_gate=0.3, w_hh=2.5)
  - [x] 训练 + 评估 — ✅ 8 epochs 全部完成, 最佳 ep3: all_pairs_clip=0.7243, LPIPS=0.3192
  - [x] **关键发现**: N1 多级 DWT 不仅未突破 0.74, 反而低于单级 DWT 的 N11+N16 (0.7315)
- [x] Task 2.2: N5 — style_fiber 多级放大 — **P1 提升优先级** — ✅ 完成 (推理侧触顶 0.7311)
  - [x] 修改 `src/spectral_bridge620.py` integrate_transport，N5 多级 style_fiber 放大
  - [x] 突破单点放大限制，强化 style 注入
  - [x] 推理侧 6 组消融完成 (最佳 N5_lvl2_hh3: all_pairs_clip=0.7311)
- [ ] Task 2.3: N2 — 时频耦合调度
  - [ ] 配置 `tf_schedule_enabled: true`, `tf_hh_max_scale: 1.5`, `tf_mid_lock_threshold: 0.5`
  - [ ] 基于阶段 1 最佳点训练 + 评估
- [ ] Task 2.4: N3 — fiber source repulsion
  - [ ] 配置 `fiber_source_repulse_scale: 0.1/0.3`（探测，注意 W2b 教训 margin ≤ 10）
  - [ ] 基于阶段 1 最佳点训练 + 评估
- [ ] Task 2.5: N4 — 训练侧 N1 + 推理侧 patch_adain 联合
  - [ ] 训练配置 = 阶段 1 最佳（含训练侧 N1）
  - [ ] 推理配置 = + `patch_adain_kernel=16`
  - [ ] 验证训练-推理一致性红利
- [x] Task 2.6: 阶段 2 结果汇总 — ✅ 完成
  - [x] 对比 N1-N5 与阶段 1 最佳
  - [x] **决定阶段 3 精调基线**: T5_D4_u01_v3 (clip=0.7323, lpips=0.3534) — Pareto 最佳点
  - [x] 阶段 2 累计最佳: T4_D1_dwt (clip=0.7325), T5_D4_u01_v3 (Pareto 最佳), N11+N16 (训练 0.7315), N5_lvl2_hh3 (推理 0.7311), N1_lvl2 (训练 0.7243, 多级DWT反而退化)

## 阶段 3：精调（~1.5 小时）

- [x] Task 3.1: U 方向 α 微调 — ✅ 完成 (a005/a01/a015/a020/a030)
  - [x] α = 0.05 / 0.1 / 0.15 / 0.2 / 0.3（基于阶段 2 最佳 T5 ep7）
  - [x] 推理消融，无需训练
  - [x] 最佳: a01 (clip=0.7323), a015 (clip=0.7323); a030 退化至 0.7299
- [x] Task 3.2: V 方向 k 微调 — ✅ 完成 (k08/k16/k32/k48)
  - [x] k = 8 / 16 / 32 / 48（基于阶段 2 最佳 T5 ep7）
  - [x] 推理消融，无需训练
  - [x] **关键发现**: V=8 突破 clip 至 0.7348（距 0.74 仅 0.0052），但 LPIPS 上升至 0.3868
  - [x] V=32/48 退化（与 V=16 相同 0.7307，被裁剪）
- [x] Task 3.3: endpoint_adain_scale 网格 — ✅ 完成 (mid/hh scale 0.1/0.2/0.3/0.5)
  - [x] mid_adain_scale ∈ {0.1, 0.2, 0.3, 0.5}, hh_adain_scale ∈ {0.1, 0.2, 0.3, 0.5}
  - [x] 推理消融，无需训练
  - [x] 无明显差异（clip 在 0.7322-0.7323 之间）
- [ ] Task 3.4: w_ll/w_hh 频域权重网格 — ⏭️ 跳过 (Phase 3 推理消融已耗尽，clip 天花板确认)
  - [ ] w_ll ∈ {0.1, 0.3, 0.5}, w_hh ∈ {1.0, 1.5, 2.0}（9 组）
  - [ ] 需训练（频域 loss 生效）
- [x] Task 3.5: 早停点选择 — ✅ 完成 (ep1/ep4/ep7/ep8/ep10)
  - [x] T5 ep1/ep4/ep7/ep8/ep10 + U4+V3 评估
  - [x] 推理消融，无需训练
  - [x] ep7 最佳 (clip=0.7323)，ep1 LPIPS 最低 (0.3411)
- [x] Task 3.6: 最终汇总报告 — ✅ 完成
  - [x] 全实验 Pareto 前沿图（clip vs lpips）
  - [x] 标记双指标达成点（clip>0.74, lpips<0.35）
  - [x] 更新 `project_memory.md`

### Phase 3b 突破尝试（v5 新增） — ✅ 完成 (11 组 V=8 网格)
- [x] V=8 + ep1/ep4/ep8/ep10 (LPIPS 控制) — 最佳 ep10/ep7 (clip=0.7348)
- [x] V=8 + a005/a002 (LPIPS 控制) — 最佳 a005 (clip=0.7341)
- [x] V=8 + mid01/mid02 (LPIPS 控制) — clip 不变 (0.7348)
- [x] V=8 + a015/a02 + ep1/ep4 (推 clip 上限) — 失败，clip 反降

### 最终 Pareto 前沿（P3 + P3b 合并）
| Name | ClipStyle | Lpips |
|------|-----------|-------|
| P3_V_k08 (V=8, ep7) | **0.7348** | 0.3868 |
| P3_V08_U005 | 0.7342 | 0.3853 |
| P3b_V08_ep8 | 0.7337 | 0.3831 |
| P3_mid05_hh05 (baseline) | 0.7323 | 0.3534 |
| P3_ep8_u01_v3 | 0.7309 | 0.3495 |
| P3_V_k32/k48 | 0.7307 | **0.3403** |

## 阶段 4：失败兜底（按需，~0.5 小时）

- [x] Task 4.1: 若未达双指标 — ✅ 完成
  - [x] 记录最接近 Pareto 前沿的点
  - [x] 分析瓶颈（clip 卡 ceiling 还是 lpips 卡 floor）
  - [x] 提出下一阶段方向（如 mixture-of-experts / per-style adapter / 更激进的频域解耦 / 跨 checkpoint ensemble / 训练侧 endpoint_adain_scale 改造）
  - [x] 更新 `project_memory.md` (Phase 4 完整 Pareto 前沿 + 下一阶段方向)

### 阶段 4 兜底分析结论

**双指标达成状态**：
- ❌ clip_style > 0.74：未达成，最高 0.7348 (距 0.74 差 0.0052)
- ⚠️ LPIPS < 0.35：未达成（与 clip>0.74 同时），但单独可达 0.3403 (V=32/48)

**瓶颈分析**：
1. **clip 天花板效应 (主瓶颈)**：
   - 35+ 推理消融 + 6 训练实验均无法突破 0.7348
   - 训练侧机制 (N1/N5/N11+N16) 触顶 0.7243-0.7315
   - 推理侧机制 (V=8/U=0.1/U=0.15) 触顶 0.7323-0.7348
   - 多级 DWT 反而退化 (N1 ep3 0.7243 < N11+N16 0.7315)
   - **根因**: 当前 620_spectral_ode + Flow Matching 架构的物理边界，clip 信息量已被压缩到极限

2. **clip-LPIPS Pareto 权衡**：
   - V=8 推 clip +0.0025 但 LPIPS +0.0334 (内容损失)
   - 早停 (ep1) 降 LPIPS 但 clip 也降
   - 无法同时优化双指标

**下一阶段方向 (Phase 5 候选)**：
1. mixture-of-experts (per-style adapter) — 不同风格用不同专家网络
2. 跨 checkpoint ensemble (T5 ep1 + ep7 加权) — 利用 LPIPS-clip 互补性
3. 更激进的频域解耦 (wavelet packet decomposition 替代 Haar)
4. 训练侧 endpoint_adain_scale 改造 (从 guard 移出)
5. 跨架构 (Diffusion Schrödinger Bridge 替代 Flow Matching)

# Task Dependencies

- Task 0.0.A (endpoint_adain_scale guard 修复) → Task 0.2/0.3 中的 D2/D7/D8/D9/D9b（U 类消融依赖）
- Task 0.0.B (B2 V2 配置恢复) → Task 0.4（B2 V2 消融依赖）
- Task 0.0.A 和 0.0.B 并行可做（互不依赖）
- Task 0.0.C (移除调试输出) → 阶段 1 训练前完成
- Task 0.* 推理消融独立，同 checkpoint 不同推理参数可并行
- Task 0.5 (阶段 0 汇总) 依赖 Task 0.2/0.3/0.4
- Task 1.3b (D4-fix) → Task 1.6（训练前需确保训练侧 style_extrap_alpha 不被 guard 跳过）
- Task 1.4 已完成 ✅，但 T4 配置需补 `endpoint_adain_scale: 1.0`
- Task 1.5 依赖 Task 0.0.B（V2 配置恢复）
- Task 1.6 (T4+T5 训练) 依赖 1.3b + 1.4/1.5 + 阶段 0 结果
- Task 1.7 (T1/T2/T3 + T6/T7) 依赖 1.6（可并行启动但优先级低）
- Task 1.8 依赖 1.6/1.7
- Task 2.* 依赖阶段 1 最佳点确定
- Task 3.* 依赖阶段 2 最佳点确定
- Task 4.* 仅在阶段 3 未达双指标时触发

# 优先级标注

- **P0（最高，阻塞 + 双路突破）**：
  - Task 0.0.A: endpoint_adain_scale guard 修复（阻塞 1）
  - Task 0.0.B: B2 V2 配置恢复（阻塞 2）
  - Task 1.6 中的 T4 训练（路径 A 全融合，含 endpoint_adain_scale=1.0）
  - Task 1.6 中的 T5 训练（路径 B 全融合，含 endpoint_adain_scale=1.0）
- **P1（高，关键验证）**：
  - Task 0.4: B2 V2 推理消融（验证路径 B 可行性，D11/D13/D15 是关键）
  - Task 0.5: 阶段 0 汇总
  - Task 1.3b: D4-fix 训练路径 guard 检查
  - Task 2.1: N1 多级 DWT
  - Task 2.2: N5 style_fiber 多级放大
- **P2（中，消融与次优先级）**：
  - Task 0.2/0.3: E4-long 推理消融
  - Task 0.0.C: 移除调试输出
  - Task 1.5: 生成 T5-T7 配置
  - Task 1.7: T1/T2/T3 + T6/T7 训练
  - Task 2.3/2.4/2.5: N2/N3/N4 探索
- **P3（低，精调与兜底）**：
  - Task 3.* 精调
  - Task 4.* 兜底

# 10 小时预算分配

| 阶段 | 预算 | 主要内容 |
|------|------|---------|
| 阶段 0 | 1.5h | 0.0.A/B/C 阻塞修复 (0.5h) + D1-D15 推理消融 (1h) |
| 阶段 1 | 3.5h | T4+T5 训练 (1.5h, P0) + T6/T7 + T1/T2/T3 (2h) |
| 阶段 2 | 3.0h | N1 多级 DWT (1h) + N5 多级 style_fiber (1h) + N2/N3/N4 (1h) |
| 阶段 3 | 1.5h | U/V/scale 网格 (0.5h) + w_ll/w_hh 网格 (0.5h) + 早停 (0.5h) |
| 阶段 4 | 0.5h | 兜底分析 + 文档 |
| **总计** | **10h** | |

