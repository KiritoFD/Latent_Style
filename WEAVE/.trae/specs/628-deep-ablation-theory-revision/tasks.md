# Tasks

## 阶段 0：前置准备与基线锁定

- [x] Task 0.1: 验证 T5 ep7 checkpoint 完整性与 baseline 指标可复现
  - 路径：`exp/p4_fusion_breakout/t5_b2v2_d2_d4/epoch_0007.pt`
  - 结果：✅ checkpoint 存在 (13.31MB)，baseline 可复现：ap_clip=0.7307, ap_lpips=0.3403

- [x] Task 0.2: 扩充 `628_gen_destructive_configs.py` 的 ABLATIONS 列表
  - 新增 D19-D30（12 组 mode 切换）
  - 新增 L13-L16（4 组损失关闭）
  - 新增 E1-E24（24 组损失启用探索）
  - 新增 P7-P18（12 参数 × 多档位 = 40 组）
  - 验证：脚本语法正确，总配置数 128（D:30 + L:16 + E:24 + P:58）

- [x] Task 0.3: 在 WSL 远程环境运行扩充后的 `628_gen_destructive_configs.py`，生成全部 128 组配置到 `configs/ablations/628_destructive/`
  - 结果：✅ 128 组配置已生成（D:30 + L:16 + E:24 + P:58）
  - 验证：所有配置 resume_checkpoint 指向 T5 ep7，num_epochs=10，full_eval_each_epoch=true
  - 抽检：D19/L13/E1/P13/P18 全部正确

## 阶段 1-3：训练侧消融（D30 + L16 + E24 + P58 = 128 组，batch 执行中）

> **执行状态**：远程 batch runner PID=4888 运行中，已完成 44 组（D1-D18 + L1-L12 + P1-P6），剩余 84 组预计 ~8h 完成。
> **单组耗时**：~341s（5.7min），含 3 epoch 训练 + 3 次 full_eval
> **关键发现**：D10-D18 组件级修改在续训 3 epoch 后对 clip/lpips 无显著影响（clip≈0.7303 vs baseline 0.7307），验证"保守吸引子稳定性"；D1 架构级修改（spectral_ode_off）显著降 clip 至 0.7136，证明频域 ODE 是核心。

### 1a. 已有 D1-D18（18 组）

- [ ] Task 1.1: 执行 D1_spectral_ode_off（contract_family 改为 620_spatial_bridge）
  - 注意：strict_resume=False，因架构变化
  - 验证：ep8/9/10 full_eval 结果写入 `exp/628_ablation/destructive/D1_spectral_ode_off/full_eval/`

- [ ] Task 1.2: 执行 D2_adain_scale_0（移除 endpoint ADAIN）
  - 验证：与 I1 (adain=0, clip=0.7291, lpips=0.3878) 对比训练侧 vs 推理侧差异

- [ ] Task 1.3: 执行 D3_alpha_0（移除 style extrapolation）
  - 验证：与 I2 (alpha=0.05) 对比，确认 α=0 时是否退化为 content identity

- [ ] Task 1.4: 执行 D4_avg_pool（DWT 替换为 avg_pool）
  - 验证：与 I10 (avg_pool, lpips=0.3871) 对比训练侧 vs 推理侧差异
  - 关键理论点：验证"DWT lowpass 是内容保持核心"命题

- [ ] Task 1.5: 执行 D5_skip_clean_off + D6_skip_blur_off（skip 连接破坏）
  - 验证：skip 路径对内容保真度的贡献

- [ ] Task 1.6: 执行 D7_decoder_highpass_off + D8_residual_gain_0 + D9_no_residual_flag
  - 验证：全局 residual 路径的必要性

- [ ] Task 1.7: 执行 D10_style_gate_film_only（关键：验证 Gate Collapse 命题）
  - 关键理论点：若 film_only 与 baseline 等价 → 证明 cross-attn 已被 bypass（gate=0.05 实际等效于 0）
  - 若 film_only 显著优于 baseline → 推翻"Gate Collapse 必然性"命题

- [ ] Task 1.8: 执行 D11_affine_gamma_0 + D12_affine_beta_0（验证 affine 调制方向）
  - 关键理论点：γ 与 β 哪个对 style 注入更关键

- [ ] Task 1.9: 执行 D13_global_gate_0 + D14_tokenizer_residual_0
  - 验证：tokenizer 内部组件对 style 编码的影响

- [ ] Task 1.10: 执行 D15_sharpen_0 + D16_endpoint_high_0 + D17_skip_residual_0
  - 验证：attention sharpen / endpoint high / skip residual 的边际贡献

- [ ] Task 1.11: 执行 D18_kinetic_off
  - 验证：kinetic penalty 对训练稳定性的贡献

### 1b. 新增 D19-D30（12 组 mode 切换）

- [ ] Task 1.12: 执行 D19-D22（style_attn_mode 切换：gated_raw / relu2 / style_select / sparsemax）
  - 关键理论点：验证 attention 选择性对 style 注入的影响
  - 与 D15 (sharpen=0) 形成完整 attention 消融组

- [ ] Task 1.13: 执行 D23（endpoint_head_mode: velocity → endpoint_lowhigh）
  - 关键理论点：验证 endpoint head 的 low/high 分离机制

- [ ] Task 1.14: 执行 D24（transport_prediction_mode: velocity → endpoint）— 训练侧 XPred
  - 关键理论点：与推理消融 #1 对比，训练侧 vs 推理侧 endpoint mode 差异
  - 验证 "训练-输出不匹配" 命题

- [ ] Task 1.15: 执行 D25（training_target_projection_mode: legacy → dwt）
  - 验证：DWT 投影对训练目标的影响

- [ ] Task 1.16: 执行 D26（kinetic_penalty_mode: global_l2 → per_band）
  - 关键理论点：验证各向异性 kinetic 对频带解耦的影响

- [ ] Task 1.17: 执行 D27（terminal_swd_mode: standard → high_freq）
  - 验证：SWD 高频模式对 style 注入的影响

- [ ] Task 1.18: 执行 D28（bridge_path_mode: vertical → tri_band）
  - 关键理论点：验证 tri_band 路径对 content/style 解耦的影响
  - 与推理消融 #9 对比训练侧 vs 推理侧

- [ ] Task 1.19: 执行 D29（swd_distance_mode: cdf → squared）
  - 验证：SWD 距离度量对 style 匹配的影响

- [ ] Task 1.20: 执行 D30（t_sampling_mode: uniform_power → logit_normal）
  - 验证：时间采样分布对训练收敛的影响

- [ ] Task 1.21: 汇总 D1-D30 结果，绘制组件-指标影响矩阵（heatmap）
  - 输出：`docs/628/_phase3_destructive_D_matrix.md`

## 阶段 2：L 类损失项破坏性消融（40 组）

### 2a. 已有 L1-L12（12 组关闭）

- [ ] Task 2.1: 执行 L1_no_endpoint_content + L2_no_endpoint_style
  - 关键理论点：endpoint style loss 是 style 注入的主梯度源，移除后 clip 应大幅下降
  - 验证 "FM 主导条件" 命题

- [ ] Task 2.2: 执行 L3_no_terminal_swd + L4_no_single_step_swd
  - 关键理论点：验证 "SWD 梯度正交性" 命题
  - 与 622 命题 3 的预测 (cos(grad_SWD, v_target) = -0.024) 对比

- [ ] Task 2.3: 执行 L5_no_single_step_edge + L6_no_kinetic
  - 验证：edge / kinetic 损失对训练稳定性的贡献

- [ ] Task 2.4: 执行 L7_no_spectral_ll + L8_no_spectral_hh + L9_no_spectral_lh_hl + L10_no_spectral_all
  - 关键理论点：spectral 各子带 loss 的独立贡献，验证频域解耦理论
  - L10 (全关) 应导致 clip 大幅下降，证明频域 ODE 是核心

- [ ] Task 2.5: 执行 L11_no_swd_high_freq + L12_no_coupling_structure
  - 验证：SWD 高频投影 / OT 结构对齐的贡献

### 2b. 新增 L13-L16（4 组关闭）

- [ ] Task 2.6: 执行 L13_no_flow（w_flow=0）— **关键！**
  - 关键理论点：验证 "FM 主导条件" 命题
  - 风险：FM 是主稳定器，关闭可能导致训练发散
  - 缓解：若发散，记录发散 epoch，仍可用于理论分析

- [ ] Task 2.7: 执行 L14_no_coupling_structure_edge + L15_no_coupling_structure_hybrid_stats
  - 验证：OT 结构损失子项的独立贡献

- [ ] Task 2.8: 执行 L16_no_endpoint_aux（source_endpoint_aux + endpoint_energy_band 联合关闭）
  - 验证：辅助 endpoint 损失的贡献

### 2c. 新增 E1-E24（24 组启用探索）

- [ ] Task 2.9: 执行 E1-E6（内容保真类启用）
  - E1 w_contrast_preserve=1.0、E2 w_channel_variance=1.0、E3 w_hf_energy=1.0
  - E4 w_content_lowpass_anchor=1.0、E5 w_content_edge_anchor=1.0、E6 w_pixel_color_match=1.0
  - 验证：这些内容保真损失能否在保持 clip 同时降低 LPIPS

- [ ] Task 2.10: 执行 E7-E12（风格强化类启用）
  - E7 w_velocity_magnitude=1.0、E8 w_residual_style_direction=1.0、E9 w_style_contrastive=1.0
  - E10 w_style_energy_floor=1.0、E11 w_hsv_saturation=1.0、E12 w_output_variance=1.0
  - 验证：这些风格强化损失能否突破 clip 天花板

- [ ] Task 2.11: 执行 E13-E16（方向约束类启用）
  - E13 w_directional_cosine=1.0、E14 w_freq_split_cosine=1.0
  - E15 w_endpoint_velocity_reg=1.0、E16 w_spectral_amplitude=1.0
  - 关键理论点：E14 频段解耦 cosine 验证频域解耦理论

- [ ] Task 2.12: 执行 E17-E20（物理约束类启用）
  - E17 w_anisotropic_kinetic=1.0、E18 w_stokes_viscous=1.0
  - E19 w_curvature=1.0、E20 w_lowfreq_velocity=1.0
  - 关键理论点：E17 各向异性 kinetic 验证 FC-SB 纤维丛理论

- [ ] Task 2.13: 执行 E21-E24（正则与蒸馏类启用）
  - E21 w_attn_entropy_reg=0.5、E22 w_style_strength_reg=0.5
  - E23 w_variance_penalty=1.0、E24 w_plain_path_distill=1.0
  - 关键理论点：E21 attention entropy reg 验证 Gate Collapse 命题

- [ ] Task 2.14: 汇总 L1-L16 + E1-E24 结果，绘制损失项-指标影响矩阵
  - 输出：`docs/628/_phase4_destructive_L_matrix.md`

## 阶段 3：P 类参数扫描（36 组）

### 3a. 已有 P1-P6（18 组）

- [ ] Task 3.1: 执行 P1_adain (025/050/075) + D2 (adain=0)
  - 绘制 adain_scale vs (clip, lpips) 曲线（5 档：0/0.25/0.5/0.75/1.0）

- [ ] Task 3.2: 执行 P2_alpha (005/020/030) + D3 (alpha=0)
  - 绘制 style_extrap_alpha vs (clip, lpips) 曲线（5 档：0/0.05/0.1/0.2/0.3）

- [ ] Task 3.3: 执行 P4_wstyle (2/4/16) + L2 (wstyle=0)
  - 绘制 w_endpoint_style vs (clip, lpips) 曲线（5 档：0/2/4/8/16）

- [ ] Task 3.4: 执行 P5_wswd (2/4/16) + L4 (wswd=0)
  - 绘制 single_step_swd_weight vs (clip, lpips) 曲线（5 档：0/2/4/8/16）

- [ ] Task 3.5: 执行 P6_gate_init (0/001/03)
  - 关键理论点：验证 "Gate Collapse 必然性"
  - 绘制 gate_init vs ep10 实际 gate 值曲线

### 3b. 新增 P7-P18（18 组权重扫描）

- [ ] Task 3.6: 执行 P7_spectral_w_hh (0.5/1.0/3.0/6.0) + L8 (w_hh=0)
  - 绘制 spectral_w_hh vs (clip, lpips) 曲线（5 档）
  - 验证：高频权重饱和点

- [ ] Task 3.7: 执行 P8_spectral_w_ll (0.1/0.5/1.0/2.0) + L7 (w_ll=0)
  - 绘制 spectral_w_ll vs (clip, lpips) 曲线（5 档）
  - 关键理论点：低频权重对 LPIPS 的影响（BASE LOCKING 理论）

- [ ] Task 3.8: 执行 P9_terminal_swd (0.05/0.5/1.0/2.0) + L3 (terminal_swd=0)
  - 绘制 terminal_swd_weight vs (clip, lpips) 曲线（5 档）

- [ ] Task 3.9: 执行 P10_w_kinetic (0.5/2.0/4.0/8.0) + L6 (w_kinetic=0)
  - 绘制 w_kinetic vs (clip, lpips) 曲线（5 档）
  - 验证：kinetic 权重饱和点

- [ ] Task 3.10: 执行 P11_bridge_sigma (0.0/0.05/0.08/0.1) + baseline (sigma=0.02)
  - 绘制 bridge_sigma vs (clip, lpips) 曲线（5 档）
  - 关键理论点：验证 σ=0.08 "魔法阈值"（训练侧）

- [ ] Task 3.11: 执行 P12_edge_weight (0.05/0.5/1.0/2.0) + L5 (edge=0)
  - 绘制 single_step_edge_weight vs (clip, lpips) 曲线（5 档）

- [ ] Task 3.12: 执行 P13_w_flow (0.1/0.3/0.5/2.0) + L13 (w_flow=0) — **关键！**
  - 绘制 w_flow vs (clip, lpips) 曲线（6 档：0/0.1/0.3/0.5/1.0/2.0）
  - 关键理论点：验证 "降低 FM 权重能否突破 clip 天花板" 假设
  - 预期：w_flow↓ → clip↑（FM 主导减弱）但 lpips↑（content 保真下降）

- [ ] Task 3.13: 执行 P14_w_endpoint_content (0.5/2.0/4.0/8.0) + L1 (w_content=0)
  - 绘制 w_endpoint_content vs (clip, lpips) 曲线（6 档）

- [ ] Task 3.14: 执行 P15_coupling_structure (0.5/2.0/4.0/8.0) + L12 (coupling=0)
  - 绘制 coupling_structure_cost_weight vs (clip, lpips) 曲线（6 档）

- [ ] Task 3.15: 执行 P16_num_tokens (64/128/512/1024) + baseline (256)
  - 绘制 style_attn_num_tokens vs (clip, lpips) 曲线（5 档）
  - 关键理论点：验证 "有效 style 维度极低" 命题（预期 256→512 边际递减）

- [ ] Task 3.16: 执行 P17_sharpen_scale (0/2.5/5.0/10.0) + D15 (sharpen=0)
  - 绘制 style_attn_sharpen_scale vs (clip, lpips) 曲线（5 档）
  - 验证：sharpen 饱和点

- [ ] Task 3.17: 执行 P18_gate_init_ext (0.1/0.5/1.0) + P6 (0/0.01/0.3)
  - 绘制 style_cross_attn_gate_init vs (clip, lpips, ep10 实际 gate) 曲线（6 档）
  - 关键理论点：验证大 gate 能否突破 Gate Collapse

- [ ] Task 3.18: 汇总 P1-P18 曲线，绘制参数-指标敏感性图
  - 输出：`docs/628/_phase5_param_sweep.md`

## 阶段 4：推理侧未覆盖消融补全（12 组）

- [ ] Task 4.1: 执行推理消融 #1 — transport_prediction_mode 切换 (velocity → endpoint)
  - 关键理论点：验证 "训练-输出不匹配" 命题
  - 与 D24 训练侧 endpoint mode 对比

- [ ] Task 4.2: 执行推理消融 #2 — style_attn_sharpen_scale (0 / 2.5 / 5.0)
  - 与 P17 训练侧 sharpen 对比

- [ ] Task 4.3: 执行推理消融 #3 — endpoint_high_scale (0 / 1.0 / 2.0)
  - 与 D16 训练侧 endpoint_high=0 对比

- [ ] Task 4.4: 执行推理消融 #4 — affine_connection_gamma_scale (0 / 0.5 / 1.0)
  - 与 D11 训练侧 gamma=0 对比

- [ ] Task 4.5: 执行推理消融 #5 — affine_connection_beta_scale (0 / 1.0 / 2.0)
  - 与 D12 训练侧 beta=0 对比

- [ ] Task 4.6: 执行推理消融 #6 — endpoint_film_init_std 推理放大 (0.02 → 0.1)
  - 验证：FiLM 强度对 style 注入的影响

- [ ] Task 4.7: 执行推理消融 #7 — style_attn_num_tokens (64 / 256 / 512)
  - 与 P16 训练侧 num_tokens 对比
  - 关键理论点：验证 "有效 style 维度极低" 命题

- [ ] Task 4.8: 执行推理消融 #8 — solver_stochastic_noise_scale (0 / 0.01 / 0.08)
  - 关键理论点：验证 FC-SB 理论中 σ=0.08 "魔法阈值"
  - 与 P11 训练侧 bridge_sigma 对比

- [ ] Task 4.9: 执行推理消融 #9 — bridge_path_mode (vertical → tri_band)
  - 与 D28 训练侧 tri_band 对比

- [ ] Task 4.10: 执行推理消融 #10 — swd_distance_mode (cdf → squared)
  - 与 D29 训练侧 squared 对比

- [ ] Task 4.11: 执行推理消融 #11 — full_eval_num_steps (4 / 8 / 16 / 32)
  - 绘制步数 vs (clip, lpips, 推理时间) 曲线
  - 验证 ODE 步数饱和点

- [ ] Task 4.12: 执行推理消融 #12 — full_eval_style_strength (0.5 / 1.0 / 1.5 / 2.0)
  - 绘制风格强度 vs (clip, lpips) 曲线
  - 验证风格强度饱和点

- [ ] Task 4.13: 汇总推理消融结果
  - 输出：`docs/628/_phase6_infer_supplement.md`

## 阶段 5：理论修正与文档更新

- [ ] Task 5.1: 校准"五层乘积保守机制"各因子
  - 基于 D10 (film_only) + P6 + P18 校准 α_gate
  - 基于 D11/D12 (affine γ/β=0) + E1/E2 校准 α_norm
  - 基于 P6 + P18 + E21 校准 α_init
  - 基于 D4 (avg_pool) 校准 α_proj
  - 基于 D19-D22 (attn mode) 校准 α_attn
  - 输出：更新后的 α_total 预测 vs 实测对比表

- [ ] Task 5.2: 验证 7 个数学命题
  - 命题 1 (Gate Collapse): D10 + P6 + P18 + E21 结果
  - 命题 2 (GN 白化): D11/D12 + E1/E2 结果
  - 命题 3 (SWD 正交): L3/L4/L11 + P9 + E24 结果
  - 命题 4 (训练-输出不匹配): D24 + 推理消融 #1 结果
  - 命题 5 (低维 style): P6 + P16 + P18 + 推理消融 #7 结果
  - 命题 6 (三难困境): 全部 114 组结果汇总
  - 命题 7 (FM 主导，新增): L13 + P13 + E20 结果
  - 每个命题标注：✅ 验证 / ❌ 推翻 / ⚠️ 部分修正

- [ ] Task 5.3: 更新 `docs/628/ablation_conclusions.md`
  - 追加 Phase 3-7 章节（D 矩阵 / L+E 矩阵 / P 曲线 / 推理补全 / 理论修正）
  - 更新 "Grand Conclusion" 章节，纳入新发现

- [ ] Task 5.4: 更新 `docs/622/history/10_unified_mathematical_model.md`
  - 追加 "7. 628 验证结果与命题修订" 章节
  - 修正 "五层乘积保守机制" 的定量值
  - 标注被推翻的命题与修正方向
  - 新增 "命题 7 (FM 主导)" 的验证结论

- [ ] Task 5.5: 更新 `docs/620/fog/theory/trivial_solution_unified.md`
  - 追加 "8. 628 破坏性消融验证" 章节
  - 修正 "平凡解流形" 的边界条件

- [ ] Task 5.6: 更新 `project_memory.md`
  - 在 "Lessons Learned" 追加 628 深度消融结论
  - 在 "Phase 5 候选方向" 中标注已被消融证伪的方向
  - 新增有效损失项清单（E 系列中 ✅ 的）

## 阶段 6：最终汇总与帕累托前沿更新

- [ ] Task 6.1: 绘制 628 完整 Pareto 前沿（含 114 组新点 + 历史 Pareto）
  - 输出：`docs/628/_phase7_pareto_frontier.md` + ASCII 图
  - 标注新 Pareto 最优点（若有）

- [ ] Task 6.2: 撰写 "Phase 5 理论指引" — 哪些方向物理上无效，哪些值得继续
  - 基于 5 层乘积模型校准结果，给出 ROI 排序
  - 标注 E 系列中值得继续探索的损失项

# Task Dependencies

- Task 0.1, 0.2, 0.3 必须先完成（前置）
- 阶段 1 (D), 阶段 2 (L+E), 阶段 3 (P) 可并行执行（共享 GPU 时分片）
- 阶段 4 (推理消融) 可与阶段 1-3 并行（不占用训练 GPU）
- 阶段 5 (理论修正) 依赖阶段 1-4 全部完成
- 阶段 6 依赖阶段 5 完成

# 执行批次规划

由于 114 组训练消融总时间 ~15h，分 3 批执行：
- **批次 1**（~5h）：D1-D18 + L1-L12（30 组，已有配置，立即可跑）
- **批次 2**（~5h）：D19-D30 + L13-L16 + E1-E24（40 组，新增配置）
- **批次 3**（~5h）：P1-P18（36 组权重扫描）+ 推理消融 #1-#12（12 组，并行）

# 风险与缓解

1. **VRAM 风险**: D1 (spectral_ode_off → spatial_bridge) 可能改变参数量导致 OOM
   - 缓解：先做 1 epoch smoke test，若 OOM 则 batch_size 降至 8

2. **strict_resume 失败**: D1/D10/D19-D22 等架构/mode 变化可能导致权重加载失败
   - 缓解：配置中已设 `strict_resume=False`，忽略缺失/多余键

3. **训练发散**: L13 (w_flow=0) 可能导致训练不稳定
   - 缓解：记录发散 epoch，发散结果仍可用于理论分析

4. **E 系列无效**: 大部分 E 系列损失项可能无显著效果
   - 缓解：这正是探索目的，"无效果"本身是理论修正的重要数据

5. **时间超预算**: 114 组可能超 15h
   - 缓解：分 3 批跨 2 天执行，推理消融并行不占训练时间

6. **fortrl error (200)**: 远程 schtask 会话 console 关闭
   - 缓解：使用 `Start-Process -RedirectStandardError -WindowStyle Hidden` wrapper
