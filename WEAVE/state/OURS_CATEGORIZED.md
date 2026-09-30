# Consolidated Result Manifest -- grounded in I: scan (2026-07-09 17:39)
Source: _images_scan_out.txt (DIR|IMG=<direct-img-count>|<path>). Only dirs with IMG>=500 listed as *complete* results.

## 1. Baselines (by protocol bucket)
### P2A-256 (256px)  [each 750]   (count=7)
- I:\exp_256_photo2art\adain_256\images  (IMG=750)
- I:\exp_256_photo2art\identity_256\images  (IMG=750)
- I:\exp_256_photo2art\samam_256\images  (IMG=750)
- I:\exp_256_photo2art\samst_256\images  (IMG=750)
- I:\exp_256_photo2art\sdturbo_256\images  (IMG=750)
- I:\exp_256_photo2art\styleid_256\images  (IMG=750)
- I:\exp_256_photo2art\wct_256\images  (IMG=750)

### Distinct5-512 SaMST / Seedream4.5  [750/721]   (count=11)
- I:\Github\Latent_Style\exp_baselines\_auxiliary_runs\cut_5x5\infer_5x5\images  (IMG=2427)
- I:\Github\Latent_Style\exp_baselines\samst_distinct5_512_wsl_stepalign40_remote_20260605_r1\eval_bundle\eval_step_000040_full\step_000040_full\images  (IMG=750)
- I:\Github\Latent_Style\exp_baselines\samst_distinct5_512_wsl_stepalign40_remote_20260605_r1\eval_bundle\eval_step_000040\step_000040\images  (IMG=750)
- I:\Github\Latent_Style\exp_baselines\samst_latent_distinct5_512_convergence_20260606_180529\eval_bundle_fast\batch300_fast\images  (IMG=750)
- I:\Github\Latent_Style\exp_baselines\samst_latent_distinct5_512_convergence_20260606_180529\eval_bundle_fast\batch950_fast\images  (IMG=750)
- I:\Github\Latent_Style\exp_baselines\samst_latent_distinct5_512_convergence_20260606_214051\eval_bundle_fast\batch1050_fast\images  (IMG=750)
- I:\Github\Latent_Style\exp_baselines\samst_latent_distinct5_512_samecost_20260606_041227\eval_bundle_fast\batch050_fast\images  (IMG=750)
- I:\Github\Latent_Style\exp_baselines\samst_latent_distinct5_512_samecost_20260606_041227\eval_bundle_fast\batch300_fast\images  (IMG=750)
- I:\Github\Latent_Style\exp_baselines\samst_latent_distinct5_512_samecost_20260606_172021\eval_bundle_fast\batch050_fast\images  (IMG=750)
- I:\Github\Latent_Style\exp_baselines\samst_latent_distinct5_512_samecost_20260606_172021\eval_bundle_fast\batch150_fast\images  (IMG=750)
- I:\Github\Latent_Style\exp_baselines\seedream45_api\distinct5_512_seedream45_windhub_20260607_repaired750\images  (IMG=750)

### Ours main eval  [750 / 444(256)]   (count=1)
- I:\exp_our_models_eval\latent512_e7\images  (IMG=750)

### Ours samst-latent eval  [750]   (count=1)
- I:\exp_samst_latent_eval\step_000001\images  (IMG=750)

### Seedream4.5 api  [750/721]   (count=1)
- I:\Github\Latent_Style\seedream45_api\protocol_a_800\images  (IMG=721)

### Gaps -- searched in I: scan, NO 750-img result dir found
- CUT / S2WAT / StyleID / SD-Turbo / SDEdit (x4): only `protocol_a_800` 1-img placeholders exist under exp_baselines; no IMG>=500 result dir. (Note: CUT 512 outputs may instead live in `_auxiliary_runs\cut_5x5\infer_5x5\images`, IMG=2427.)
- StyleAligned / Z-STAR / StyleShot: not present in I: scan at all (not run).
- AdaIN / Identity / WCT / SaMam 512: not in exp_baselines; their 256px versions are in P2A-256 bucket above.

## 2. Ours experiments in experiments_historical  (total image-dirs=447, distinct experiments=175)

### [Ablate-patch/id/tv]  --  12 image-dirs, 8 distinct experiments
  - ablate_A0_base_p5_id045_tv005  (dirs=1)
  - ablate_A1_p7_id045_tv005  (dirs=1)
  - ablate_A2_p11_id045_tv005  (dirs=1)
  - ablate_A3_p5_id030_tv005  (dirs=1)
  - ablate_A4_p5_id070_tv005  (dirs=1)
  - ablate_A5_p5_id045_tv003  (dirs=1)
  - ablate_M1-Aggressive-Fine  (dirs=3)
  - ablate_M2-Smooth-Impasto  (dirs=3)

### [AdaMix]  --  1 image-dirs, 1 distinct experiments
  - adamix  (dirs=1)

### [Aline-Albedo-120]  --  34 image-dirs, 5 distinct experiments
  - Aline120_aline_01_oracle  (dirs=6)
  - Aline120_aline_02_texture_maniac  (dirs=6)
  - Aline120_aline_03_ghost_wireframe  (dirs=10)
  - Aline120_aline_04_macro_trap  (dirs=6)
  - Aline120_aline_05_idt_poison  (dirs=6)

### [Arch-ablation]  --  36 image-dirs, 18 distinct experiments
  - arch_1_pM_sC_dH  (dirs=2)
  - arch_2_pM_sA_dL  (dirs=2)
  - arch_3_pM_sC_dL  (dirs=2)
  - arch_4_pM_sA_dH  (dirs=2)
  - arch_5_pMW_sA_dH  (dirs=2)
  - arch_6_pMW_sC_dL  (dirs=2)
  - arch_7_pMW_sA_dL  (dirs=2)
  - arch_8_pMW_sC_dH  (dirs=2)
  - arch_ablate_A1_swin_h2_g1_d2  (dirs=2)
  - arch_ablate_A2_swin_h2_g2_d2  (dirs=2)
  - arch_ablate_A3_swin_h3_g2_d2  (dirs=2)
  - arch_ablate_B1_weaver_h3_g2_d1  (dirs=2)
  - arch_ablate_B2_weaver_h2_g2_d1  (dirs=2)
  - arch_ablate_B3_weaver_h3_g2_d2  (dirs=2)
  - arch_ablate_C1_asym_h1_g2_d2  (dirs=2)
  - arch_ablate_C2_asym_h2_g2_d3  (dirs=2)
  - arch_ablate_D1_cgw_h2_g2_d3_impasto_s3_r12  (dirs=2)
  - arch_ablate_E1_wgw_light_h2_g1_d2  (dirs=2)

### [Arch-Family-42]  --  46 image-dirs, 10 distinct experiments
  - 42_A01_Macro_Only_LR3e4  (dirs=6)
  - 42_A02_Micro_Only_LR3e4  (dirs=6)
  - 42_A03_Bipolar_Extreme  (dirs=6)
  - 42_A04_FullSpec_Conv2_LR4e4  (dirs=4)
  - 42_A05_FullSpec_Conv3_LR2e4  (dirs=4)
  - 42_A06_FullSpec_Conv3_LR3e4  (dirs=4)
  - 42_A07_NoSkip_Conv2_LR3e4  (dirs=4)
  - 42_A08_Noise01_Conv2_LR3e4  (dirs=4)
  - 42_A09_Gain4_Conv2_LR2e4  (dirs=4)
  - 42_A10_Color200_Conv2_LR3e4  (dirs=4)

### [Arch-Family-45]  --  8 image-dirs, 4 distinct experiments
  - 45_01_golden_funnel  (dirs=2)
  - 45_02_naked_fusion  (dirs=2)
  - 45_03_macro_dictator  (dirs=2)
  - 45_04_micro_rebel  (dirs=2)

### [CA-parameter]  --  24 image-dirs, 12 distinct experiments
  - ca_pram_final_1_lr4_id35_swd60_c5  (dirs=2)
  - ca_pram_final_10_dim96_tok128  (dirs=2)
  - ca_pram_final_11_dim128_tok128  (dirs=2)
  - ca_pram_final_12_base_ref  (dirs=2)
  - ca_pram_final_2_lr5_id30_swd80_c2  (dirs=2)
  - ca_pram_final_3_lr6_id25_swd60_c5  (dirs=2)
  - ca_pram_final_4_lr8_id30_swd80_c2  (dirs=2)
  - ca_pram_final_5_lr5_id15_swd120_c5  (dirs=2)
  - ca_pram_final_6_lr5_id20_swd150_c10  (dirs=2)
  - ca_pram_final_7_lr8_id15_swd120_c5  (dirs=2)
  - ca_pram_final_8_lr8_id20_swd150_c10  (dirs=2)
  - ca_pram_final_9_dim128_tok64  (dirs=2)

### [Color-120]  --  12 image-dirs, 2 distinct experiments
  - Color120_C01_HF_Tyrant  (dirs=6)
  - Color120_C02_HF_Leakage  (dirs=6)

### [ColorCorrelation]  --  10 image-dirs, 5 distinct experiments
  - clocor1_E1_Macro19_Rigid_LR14e4  (dirs=2)
  - clocor1_E2_15Series_Rigid_LR14e4  (dirs=2)
  - clocor1_E3_15Series_Soft_LR14e4  (dirs=2)
  - clocor1_E4_9Series_Rigid_LR14e4  (dirs=2)
  - clocor1_E5_9Series_Soft_LR14e4  (dirs=2)

### [Component-Ablate-43]  --  34 image-dirs, 17 distinct experiments
  - Ablate43_A01_ResOn_TheFilter  (dirs=2)
  - Ablate43_A02_Capacity_Conv1  (dirs=2)
  - Ablate43_A03_WindowAttn_Size8  (dirs=2)
  - Ablate43_A04_Modulator_GlobalOnly  (dirs=2)
  - Ablate43_I01_Skip_TotalBlind  (dirs=2)
  - Ablate43_I02_Skip_ConcatFusion  (dirs=2)
  - Ablate43_I03_Gate_Hires_Only  (dirs=2)
  - Ablate43_I04_Gain_Vanilla  (dirs=2)
  - Ablate43_L01_SWD_TurnOff  (dirs=2)
  - Ablate43_L02_Color_TurnOff  (dirs=2)
  - Ablate43_L03_IDT_MassiveReturn  (dirs=2)
  - Ablate43_L04_SWD_Nuke  (dirs=2)
  - Ablate43_P01_Patch_LargeOnly  (dirs=2)
  - Ablate43_P02_Patch_FullSpectrum  (dirs=2)
  - Ablate43_P03_Patch_NanoClash  (dirs=2)
  - Ablate43_S01_Baseline_Gold  (dirs=2)
  - Ablate43_S02_DeepConv3  (dirs=2)

### [CrossAttention]  --  14 image-dirs, 6 distinct experiments
  - cross_attn_Run_0_Baseline  (dirs=1)
  - cross_attn_Run_1_lr_high_8e4  (dirs=1)
  - cross_attn_v3_v3_0_attn_base  (dirs=3)
  - cross_attn_v3_v3_1_arch_dict  (dirs=3)
  - cross_attn_v3_v3_2_skip_naive  (dirs=3)
  - cross_attn_v3_v3_3_no_residual  (dirs=3)

### [Decoder-variants]  --  22 image-dirs, 12 distinct experiments
  - 1-decoder-no_norm-patch5_23-color1.0  (dirs=1)
  - 1-decoder-patch5-15  (dirs=2)
  - decoder-1  (dirs=2)
  - decoder-A-anchor-nohf  (dirs=2)
  - decoder-B-hf-strict-id  (dirs=2)
  - decoder-C-relaxed-id-nohf  (dirs=2)
  - decoder-D-sweetspot  (dirs=2)
  - decoder-E-extreme-brush  (dirs=2)
  - decoder-H-MSCTM  (dirs=2)
  - decoder-H-MSCTM-idt_1-tv_0.3  (dirs=1)
  - decoder-H-MSCTM-mult-tv-1  (dirs=1)
  - decoder-H-MSCTM-no_clamp_mult-tv-2  (dirs=3)

### [Delta]  --  2 image-dirs, 2 distinct experiments
  - delta_A0_base_p5_id045_tv005  (dirs=1)
  - delta_A1_p7_id045_tv005  (dirs=1)

### [Demodulation]  --  2 image-dirs, 1 distinct experiments
  - final_demodulation  (dirs=2)

### [Dict/Codebook]  --  1 image-dirs, 1 distinct experiments
  - dict  (dirs=1)

### [Disabled-component]  --  6 image-dirs, 3 distinct experiments
  - no-dict-hf-swd  (dirs=1)
  - no-tv  (dirs=1)
  - nstyle-proj  (dirs=4)

### [ExpS-zero-id/color-blind]  --  2 image-dirs, 2 distinct experiments
  - exp_S1_zero_id  (dirs=1)
  - exp_S2_color_blind  (dirs=1)

### [Final-Micro-2]  --  8 image-dirs, 2 distinct experiments
  - FinalMicro_2_F01_Patch135_Gain1.5_LR2e4  (dirs=4)
  - FinalMicro_2_F02_Patch357_Gain1.5_LR2e4  (dirs=4)

### [Freq-Domain]  --  36 image-dirs, 9 distinct experiments
  - freq_01_conservative_baseline  (dirs=4)
  - freq_02_brush_frenzy  (dirs=4)
  - freq_03_large_view_awareness  (dirs=4)
  - freq_04_no_idt_abyss  (dirs=4)
  - freq_05_idt_iron_fist  (dirs=4)
  - freq_06_yuv_dictatorship  (dirs=4)
  - freq_07_remove_blast_wall  (dirs=4)
  - freq_08_extreme_asymmetry  (dirs=4)
  - freq_09_lancet  (dirs=4)

### [Gain]  --  9 image-dirs, 4 distinct experiments
  - G0_Balanced_Base  (dirs=3)
  - G0-Base-Gain0.5  (dirs=2)
  - G1_High_HF_Test  (dirs=3)
  - G1-Relax-ID  (dirs=1)

### [HeavyDecode]  --  2 image-dirs, 1 distinct experiments
  - heavy_decode  (dirs=2)

### [Injection]  --  15 image-dirs, 8 distinct experiments
  - inject_I0_all_open  (dirs=2)
  - inject_I1_body_only  (dirs=2)
  - inject_I2_hires_decoder_only  (dirs=2)
  - inject_I3_progressive_1_05_01  (dirs=2)
  - inject_I4_body_hires  (dirs=2)
  - inject_I5_body_decoder  (dirs=2)
  - inject_I6_hires_only  (dirs=2)
  - inject_I7_decoder_only  (dirs=1)

### [LayerNorm-idt-sched]  --  1 image-dirs, 1 distinct experiments
  - Layer-Norm-idt_schedule  (dirs=1)

### [Light]  --  2 image-dirs, 2 distinct experiments
  - light-1  (dirs=1)
  - light-15patch-10color  (dirs=1)

### [Loss-weight]  --  14 image-dirs, 7 distinct experiments
  - weight_0_base  (dirs=2)
  - weight_1_swd_low  (dirs=2)
  - weight_2_swd_high  (dirs=2)
  - weight_3_color_low  (dirs=2)
  - weight_4_color_high  (dirs=2)
  - weight_5_id_loose  (dirs=2)
  - weight_6_id_tight  (dirs=2)

### [Micro-E]  --  16 image-dirs, 4 distinct experiments
  - micro_E01_Patch3_Gain4_LR2e4  (dirs=4)
  - micro_E02_Patch3_5_Gain4_LR2e4  (dirs=4)
  - micro_E03_Patch3_5_7_Gain4_LR2e4  (dirs=4)
  - micro_E04_Patch1_3_5_Gain4_LR2e4  (dirs=4)

### [NCE-loss]  --  13 image-dirs, 5 distinct experiments
  - nce  (dirs=4)
  - nce-gate_content  (dirs=2)
  - nce-gate_norm  (dirs=1)
  - nce-gate_norm-swd_0.45-cl_0.01  (dirs=3)
  - nce-swd_0.25-cl_0.01  (dirs=3)

### [New4-stylized]  --  8 image-dirs, 4 distinct experiments
  - New4_N01_Stylized_Naive  (dirs=2)
  - New4_N02_Stylized_Adaptive  (dirs=2)
  - New4_N03_Stylized_Adaptive_Retain0p2  (dirs=2)
  - New4_N04_Stylized_Norm  (dirs=2)

### [Optuna-HPO-style_oa]  --  28 image-dirs, 1 distinct experiments
  - experiment  (dirs=28)

### [Patch]  --  14 image-dirs, 7 distinct experiments
  - p_1_5_9_15_hf_1p0  (dirs=2)
  - p_1_5_9_15_hf_3p0  (dirs=2)
  - p_1_5_9_15_hf_off  (dirs=2)
  - p_5_9_15_25_hf_off  (dirs=2)
  - p_base_hf_1p0  (dirs=2)
  - p_base_hf_3p0  (dirs=2)
  - p_base_hf_off  (dirs=2)

### [Skip-connection]  --  9 image-dirs, 5 distinct experiments
  - Skip10_S01_None  (dirs=2)
  - Skip10_S02_Naive_G1p0  (dirs=2)
  - Skip10_S03_Naive_G0p5  (dirs=2)
  - Skip10_S04_Naive_G1p5  (dirs=2)
  - Skip10_S05_Adaptive  (dirs=1)

### [Strong-IDT]  --  6 image-dirs, 6 distinct experiments
  - strong_idt30_swd100_color20  (dirs=1)
  - strong_idt30_swd100_color50  (dirs=1)
  - strong_idt30_swd100_color80  (dirs=1)
  - strong_idt30_swd150_color20  (dirs=1)
  - strong_idt30_swd150_color50  (dirs=1)
  - strong_idt30_swd150_color80  (dirs=1)
