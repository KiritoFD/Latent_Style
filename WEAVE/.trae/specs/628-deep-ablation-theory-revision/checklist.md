# Checklist

## 阶段 0：前置准备
- [ ] T5 ep7 checkpoint 存在且 baseline 指标可复现 (clip=0.7307, lpips=0.3403)
- [ ] `628_gen_destructive_configs.py` ABLATIONS 列表已扩充（D19-D30 + L13-L16 + E1-E24 + P7-P18）
- [ ] `configs/ablations/628_destructive/` 目录下 ~114 组配置文件齐全
- [ ] 每组配置的 `resume_checkpoint` 指向 T5 ep7
- [ ] 每组配置的 `num_epochs=10`, `full_eval_each_epoch=true`, `save_interval=1`
- [ ] Windows/WSL 路径正确（I: 盘 vs /mnt/i/）

## 阶段 1：D 类架构消融（30 组）

### 1a. D1-D18（已有）
- [ ] D1_spectral_ode_off: ep8/9/10 full_eval 完整，结果可读
- [ ] D2_adain_scale_0: 结果与 I1 (adain=0 推理) 对比记录
- [ ] D3_alpha_0: 结果与 I2 (alpha=0.05 推理) 对比记录
- [ ] D4_avg_pool: 结果与 I10 (avg_pool 推理) 对比记录
- [ ] D5_skip_clean_off, D6_skip_blur_off: 结果完整
- [ ] D7_decoder_highpass_off, D8_residual_gain_0, D9_no_residual_flag: 结果完整
- [ ] D10_style_gate_film_only: 结果用于验证 Gate Collapse 命题
- [ ] D11_affine_gamma_0, D12_affine_beta_0: 结果用于校准 α_norm
- [ ] D13_global_gate_0, D14_tokenizer_residual_0: 结果完整
- [ ] D15_sharpen_0, D16_endpoint_high_0, D17_skip_residual_0: 结果完整
- [ ] D18_kinetic_off: 结果完整

### 1b. D19-D30（新增 mode 切换）
- [ ] D19 style_attn_mode=gated_raw: 结果完整
- [ ] D20 style_attn_mode=relu2: 结果完整
- [ ] D21 style_attn_mode=style_select: 结果完整
- [ ] D22 style_attn_mode=sparsemax: 结果完整
- [ ] D23 endpoint_head_mode=endpoint_lowhigh: 结果完整
- [ ] D24 transport_prediction_mode=endpoint: 结果用于验证训练-输出不匹配命题
- [ ] D25 training_target_projection_mode=dwt: 结果完整
- [ ] D26 kinetic_penalty_mode=per_band: 结果完整
- [ ] D27 terminal_swd_mode=high_freq: 结果完整
- [ ] D28 bridge_path_mode=tri_band: 结果完整
- [ ] D29 swd_distance_mode=squared: 结果完整
- [ ] D30 t_sampling_mode=logit_normal: 结果完整
- [ ] D 矩阵 heatmap 文档生成: `docs/628/_phase3_destructive_D_matrix.md`

## 阶段 2：L 类损失消融（40 组）

### 2a. L1-L12（已有关闭）
- [ ] L1_no_endpoint_content, L2_no_endpoint_style: 结果完整
- [ ] L3_no_terminal_swd, L4_no_single_step_swd: 结果用于验证 SWD 正交性命题
- [ ] L5_no_single_step_edge, L6_no_kinetic: 结果完整
- [ ] L7_no_spectral_ll, L8_no_spectral_hh, L9_no_spectral_lh_hl, L10_no_spectral_all: 结果完整
- [ ] L11_no_swd_high_freq, L12_no_coupling_structure: 结果完整

### 2b. L13-L16（新增关闭）
- [ ] L13_no_flow (w_flow=0): 结果完整，用于验证 FM 主导命题
- [ ] L14_no_coupling_structure_edge: 结果完整
- [ ] L15_no_coupling_structure_hybrid_stats: 结果完整
- [ ] L16_no_endpoint_aux: 结果完整

### 2c. E1-E24（新增启用探索）
- [ ] E1 w_contrast_preserve=1.0: 结果完整
- [ ] E2 w_channel_variance=1.0: 结果完整
- [ ] E3 w_hf_energy=1.0: 结果完整
- [ ] E4 w_content_lowpass_anchor=1.0: 结果完整
- [ ] E5 w_content_edge_anchor=1.0: 结果完整
- [ ] E6 w_pixel_color_match=1.0: 结果完整
- [ ] E7 w_velocity_magnitude=1.0: 结果完整
- [ ] E8 w_residual_style_direction=1.0: 结果完整
- [ ] E9 w_style_contrastive=1.0: 结果完整
- [ ] E10 w_style_energy_floor=1.0: 结果完整
- [ ] E11 w_hsv_saturation=1.0: 结果完整
- [ ] E12 w_output_variance=1.0: 结果完整
- [ ] E13 w_directional_cosine=1.0: 结果完整
- [ ] E14 w_freq_split_cosine=1.0: 结果完整，用于验证频域解耦理论
- [ ] E15 w_endpoint_velocity_reg=1.0: 结果完整
- [ ] E16 w_spectral_amplitude=1.0: 结果完整
- [ ] E17 w_anisotropic_kinetic=1.0: 结果完整，用于验证 FC-SB 纤维丛理论
- [ ] E18 w_stokes_viscous=1.0: 结果完整
- [ ] E19 w_curvature=1.0: 结果完整
- [ ] E20 w_lowfreq_velocity=1.0: 结果完整，用于验证 FM 主导命题
- [ ] E21 w_attn_entropy_reg=0.5: 结果完整，用于验证 Gate Collapse 命题
- [ ] E22 w_style_strength_reg=0.5: 结果完整
- [ ] E23 w_variance_penalty=1.0: 结果完整
- [ ] E24 w_plain_path_distill=1.0: 结果完整
- [ ] L+E 矩阵文档生成: `docs/628/_phase4_destructive_L_matrix.md`

## 阶段 3：P 类参数扫描（36 组）

### 3a. P1-P6（已有）
- [ ] P1_adain (0/0.25/0.5/0.75/1.0) 5 档曲线完整
- [ ] P2_alpha (0/0.05/0.1/0.2/0.3) 5 档曲线完整
- [ ] P4_wstyle (0/2/4/8/16) 5 档曲线完整
- [ ] P5_wswd (0/2/4/8/16) 5 档曲线完整
- [ ] P6_gate_init (0/0.01/0.05/0.3) 曲线，记录 ep10 实际 gate 值

### 3b. P7-P18（新增权重扫描）
- [ ] P7_spectral_w_hh (0/0.5/1.0/1.5/3.0/6.0) 曲线完整
- [ ] P8_spectral_w_ll (0/0.1/0.3/0.5/1.0/2.0) 曲线完整
- [ ] P9_terminal_swd (0/0.05/0.1/0.5/1.0/2.0) 曲线完整
- [ ] P10_w_kinetic (0/0.5/1.0/2.0/4.0/8.0) 曲线完整
- [ ] P11_bridge_sigma (0/0.02/0.05/0.08/0.1) 曲线完整
- [ ] P12_edge_weight (0/0.05/0.1/0.5/1.0/2.0) 曲线完整
- [ ] P13_w_flow (0/0.1/0.3/0.5/1.0/2.0) 曲线完整，验证 FM 权重假设
- [ ] P14_w_endpoint_content (0/0.5/1.0/2.0/4.0/8.0) 曲线完整
- [ ] P15_coupling_structure (0/0.5/1.0/2.0/4.0/8.0) 曲线完整
- [ ] P16_num_tokens (64/128/256/512/1024) 曲线完整，验证低维 style 命题
- [ ] P17_sharpen_scale (0/2.5/5.0/10.0) 曲线完整
- [ ] P18_gate_init_ext (0/0.01/0.05/0.1/0.3/0.5/1.0) 曲线完整，验证大 gate 效果
- [ ] 参数敏感性图文档生成: `docs/628/_phase5_param_sweep.md`

## 阶段 4：推理消融补全（12 组）
- [ ] 推理消融 #1 (endpoint mode): 结果记录，与 D24 训练侧对比
- [ ] 推理消融 #2 (sharpen_scale 0/2.5/5.0): 结果完整
- [ ] 推理消融 #3 (endpoint_high_scale 0/1.0/2.0): 结果完整
- [ ] 推理消融 #4 (affine_gamma 0/0.5/1.0): 结果完整
- [ ] 推理消融 #5 (affine_beta 0/1.0/2.0): 结果完整
- [ ] 推理消融 #6 (film_init_std 0.02→0.1): 结果完整
- [ ] 推理消融 #7 (num_tokens 64/256/512): 结果用于验证低维 style 命题
- [ ] 推理消融 #8 (noise_scale 0/0.01/0.08): 结果用于验证 σ=0.08 魔法阈值
- [ ] 推理消融 #9 (bridge_path_mode=tri_band): 结果完整，与 D28 对比
- [ ] 推理消融 #10 (swd_distance_mode=squared): 结果完整，与 D29 对比
- [ ] 推理消融 #11 (num_steps 4/8/16/32): 曲线完整，验证 ODE 步数饱和点
- [ ] 推理消融 #12 (style_strength 0.5/1.0/1.5/2.0): 曲线完整，验证风格强度饱和点
- [ ] 推理消融汇总文档: `docs/628/_phase6_infer_supplement.md`

## 阶段 5：理论修正
- [ ] 五层乘积保守机制各因子 (α_gate/α_attn/α_norm/α_init/α_proj) 已基于消融结果校准
- [ ] 7 个数学命题验证表完成 (每命题标注 ✅/❌/⚠️)
  - [ ] 命题 1 (Gate Collapse): D10 + P6 + P18 + E21
  - [ ] 命题 2 (GN 白化): D11/D12 + E1/E2
  - [ ] 命题 3 (SWD 正交): L3/L4/L11 + P9 + E24
  - [ ] 命题 4 (训练-输出不匹配): D24 + 推理消融 #1
  - [ ] 命题 5 (低维 style): P6 + P16 + P18 + 推理消融 #7
  - [ ] 命题 6 (三难困境): 全部 114 组汇总
  - [ ] 命题 7 (FM 主导，新增): L13 + P13 + E20
- [ ] `docs/628/ablation_conclusions.md` 追加 Phase 3-7 章节
- [ ] `docs/622/history/10_unified_mathematical_model.md` 追加 "7. 628 验证结果" 章节
- [ ] `docs/620/fog/theory/trivial_solution_unified.md` 追加 "8. 628 破坏性消融验证" 章节
- [ ] `project_memory.md` "Lessons Learned" 与 "Phase 5 候选方向" 更新
- [ ] 有效损失项清单（E 系列中 ✅ 的）记录到 project_memory

## 阶段 6：最终汇总
- [ ] 628 完整 Pareto 前沿图绘制 (含 114 组新点 + 历史点)
- [ ] "Phase 5 理论指引" 文档完成 (ROI 排序的方向建议)

## 工程约束
- [ ] 全程显存峰值 < 10.8 GB (T5 baseline 峰值 8.9 GB)
- [ ] DataLoader: num_workers=0, pin_memory=False, persistent_workers=False
- [ ] 数据集路径: I 盘 (/mnt/i/...)，非 F 盘
- [ ] Test 目录: /mnt/i/wikiart_distinct5_samam_512_classview/test
- [ ] 全部训练日志完整捕获到 exp/628_ablation/destructive/<name>/train.log
- [ ] 单组实验失败不中断后续 (if...then...else 容错)
- [ ] fortrl error (200) 已用 Start-Process -RedirectStandardError wrapper 规避
- [ ] 分 3 批执行，每批 ~5h，跨 2 天完成

## 数量核对
- [ ] D 类: 30 组 (D1-D18 已有 + D19-D30 新增)
- [ ] L 类: 16 组 (L1-L12 已有 + L13-L16 新增)
- [ ] E 类: 24 组 (E1-E24 全新)
- [ ] P 类: 36 组 (P1-P6 已有 18 组 + P7-P18 新增 18 组)
- [ ] 推理消融: 12 组 (#1-#8 已规划 + #9-#12 新增)
- [ ] 训练消融总计: 30 + 16 + 24 + 36 = 106 组
- [ ] 推理消融总计: 12 组
- [ ] 总计: 118 组 (含部分重叠点如 D2=P1_adain_0)
