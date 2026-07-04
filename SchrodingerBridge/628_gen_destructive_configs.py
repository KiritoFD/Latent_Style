"""628 Destructive Ablation: Training-side config generator.

Generates configs for D (architecture), L (loss), P (parameter sweep) ablations.
Each config resumes from T5 ep7, trains 3 new epochs (ep8-10).
Output: configs/ablations/628_destructive/

Key findings from code audit:
- spectral_ode_enabled: DEAD CODE (never read, dispatch via contract_family)
- spectral_ode_levels=0: clamped to max(1,...) → use contract_family="620_spatial_bridge" instead
- style_gate_mode="none": silently falls back to tanh_gate → use "film_only" (gate=0)
- lowpass_mode="none": silently falls back to avg_pool → use "avg_pool" explicitly
- color_highway_gain: DEAD CODE (stored but never used in forward)
- body_block_type="conv": CHANGES PARAM COUNT → breaks resume → SKIP
- skip_fusion_mode="none": invalid + changes arch → SKIP
- Inference-side ablations: style_gate_mode/film_only gives same result as baseline
  (already-trained weights just use residual path) → ALL destructive ablations must be TRAINING-side
"""
import json
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
T5_CONFIG_PATH = ROOT / "exp" / "p4_fusion_breakout" / "t5_b2v2_d2_d4" / "config.json"
OUTPUT_DIR = ROOT / "configs" / "ablations" / "628_destructive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ABLATIONS = [
    # ===== A. Architecture component ablations (D) =====
    {
        "name": "D1_spectral_ode_off",
        "desc": "Remove spectral ODE (use spatial bridge instead of spectral bridge)",
        "overrides": {
            "model.contract_family": "620_spatial_bridge",
        },
        "strict_resume": False,
    },
    {
        "name": "D2_adain_scale_0",
        "desc": "Remove endpoint ADAIN (scale=0, no style modulation at endpoint)",
        "overrides": {
            "model.endpoint_adain_scale": 0,
        },
    },
    {
        "name": "D3_alpha_0",
        "desc": "Remove style extrapolation (alpha=0, no extrapolation beyond training distribution)",
        "overrides": {
            "model.style_extrap_alpha": 0,
        },
    },
    {
        "name": "D4_avg_pool",
        "desc": "Replace DWT with avg_pool lowpass (weaker frequency decomposition)",
        "overrides": {
            "model.lowpass_mode": "avg_pool",
        },
    },
    {
        "name": "D5_skip_clean_off",
        "desc": "Disable skip clean (add noise to skip connection)",
        "overrides": {
            "model.ablation_skip_clean": False,
        },
    },
    {
        "name": "D6_skip_blur_off",
        "desc": "Disable skip blur (remove lowpass from skip connection)",
        "overrides": {
            "model.ablation_skip_blur": False,
        },
    },
    {
        "name": "D7_decoder_highpass_off",
        "desc": "Disable decoder highpass (decoder operates on full frequency)",
        "overrides": {
            "model.ablation_decoder_highpass": False,
        },
    },
    {
        "name": "D8_residual_gain_0",
        "desc": "Remove global residual (residual_gain=0, no anchor+delta path)",
        "overrides": {
            "model.residual_gain": 0,
        },
    },
    {
        "name": "D9_no_residual_flag",
        "desc": "Remove residual via ablation flag (outputs scaled raw delta)",
        "overrides": {
            "model.ablation_no_residual": True,
        },
    },
    {
        "name": "D10_style_gate_film_only",
        "desc": "Style gate=film_only (zero out cross-attention output, rely on FiLM only)",
        "overrides": {
            "model.style_gate_mode": "film_only",
        },
    },
    {
        "name": "D11_affine_gamma_0",
        "desc": "Remove affine gamma modulation (no scale modulation, gamma=0→identity)",
        "overrides": {
            "model.affine_connection_gamma_scale": 0,
        },
    },
    {
        "name": "D12_affine_beta_0",
        "desc": "Remove affine beta modulation (no shift modulation, beta=0→no shift)",
        "overrides": {
            "model.affine_connection_beta_scale": 0,
        },
    },
    {
        "name": "D13_global_gate_0",
        "desc": "Remove global gate from tokenizer (global_gate_scale=0, disable style path)",
        "overrides": {
            "model.tokenizer_global_gate_scale": 0,
        },
    },
    {
        "name": "D14_tokenizer_residual_0",
        "desc": "Remove tokenizer residual (residual_gain=0, no skip in tokenizer)",
        "overrides": {
            "model.tokenizer_residual_gain": 0,
        },
    },
    {
        "name": "D15_sharpen_0",
        "desc": "Remove style attention sharpening (sharpen_scale=0)",
        "overrides": {
            "model.style_attn_sharpen_scale": 0,
        },
    },
    {
        "name": "D16_endpoint_high_0",
        "desc": "Remove endpoint high frequency scale (high_scale=0)",
        "overrides": {
            "model.endpoint_high_scale": 0,
        },
    },
    {
        "name": "D17_skip_residual_0",
        "desc": "Remove skip residual weight (skip_residual_weight=0, no blending)",
        "overrides": {
            "model.skip_residual_weight": 0,
        },
    },
    {
        "name": "D18_kinetic_off",
        "desc": "Disable kinetic penalty mode",
        "overrides": {
            "bridge.kinetic_penalty_mode": "off",
        },
    },

    # ===== B. Loss ablations (L) =====
    {
        "name": "L1_no_endpoint_content",
        "desc": "Remove endpoint content loss (w_endpoint_content=0)",
        "overrides": {
            "bridge.w_endpoint_content": 0,
        },
    },
    {
        "name": "L2_no_endpoint_style",
        "desc": "Remove endpoint style loss (w_endpoint_style=0)",
        "overrides": {
            "bridge.w_endpoint_style": 0,
        },
    },
    {
        "name": "L3_no_terminal_swd",
        "desc": "Remove terminal SWD loss (terminal_swd_weight=0)",
        "overrides": {
            "bridge.terminal_swd_weight": 0,
        },
    },
    {
        "name": "L4_no_single_step_swd",
        "desc": "Remove single-step SWD loss (single_step_swd_weight=0)",
        "overrides": {
            "bridge.single_step_swd_weight": 0,
        },
    },
    {
        "name": "L5_no_single_step_edge",
        "desc": "Remove single-step edge loss (single_step_edge_weight=0)",
        "overrides": {
            "bridge.single_step_edge_weight": 0,
        },
    },
    {
        "name": "L6_no_kinetic",
        "desc": "Remove kinetic energy loss (w_kinetic=0)",
        "overrides": {
            "bridge.w_kinetic": 0,
        },
    },
    {
        "name": "L7_no_spectral_ll",
        "desc": "Remove spectral LL (low-freq) loss (spectral_w_ll=0)",
        "overrides": {
            "bridge.spectral_w_ll": 0,
        },
    },
    {
        "name": "L8_no_spectral_hh",
        "desc": "Remove spectral HH (texture) loss (spectral_w_hh=0)",
        "overrides": {
            "bridge.spectral_w_hh": 0,
        },
    },
    {
        "name": "L9_no_spectral_lh_hl",
        "desc": "Remove spectral LH+HL (directional detail) loss",
        "overrides": {
            "bridge.spectral_w_lh": 0,
            "bridge.spectral_w_hl": 0,
        },
    },
    {
        "name": "L10_no_spectral_all",
        "desc": "Remove ALL spectral FM losses (ll+lh+hl+hh=0)",
        "overrides": {
            "bridge.spectral_w_ll": 0,
            "bridge.spectral_w_lh": 0,
            "bridge.spectral_w_hl": 0,
            "bridge.spectral_w_hh": 0,
        },
    },
    {
        "name": "L11_no_swd_high_freq",
        "desc": "Disable SWD high frequency projection (swd_use_high_freq=False)",
        "overrides": {
            "bridge.swd_use_high_freq": False,
        },
    },
    {
        "name": "L12_no_coupling_structure",
        "desc": "Remove coupling structure cost (coupling_structure_cost_weight=0)",
        "overrides": {
            "bridge.coupling_structure_cost_weight": 0,
        },
    },

    # ===== C. Parameter sweeps (P) =====
    # P1: endpoint_adain_scale sweep (includes D2 at value=0)
    {
        "name": "P1_adain_025",
        "desc": "endpoint_adain_scale=0.25",
        "overrides": {"model.endpoint_adain_scale": 0.25},
    },
    {
        "name": "P1_adain_050",
        "desc": "endpoint_adain_scale=0.50",
        "overrides": {"model.endpoint_adain_scale": 0.5},
    },
    {
        "name": "P1_adain_075",
        "desc": "endpoint_adain_scale=0.75",
        "overrides": {"model.endpoint_adain_scale": 0.75},
    },
    # P2: style_extrap_alpha sweep (includes D3 at value=0)
    {
        "name": "P2_alpha_005",
        "desc": "style_extrap_alpha=0.05",
        "overrides": {"model.style_extrap_alpha": 0.05},
    },
    {
        "name": "P2_alpha_020",
        "desc": "style_extrap_alpha=0.20",
        "overrides": {"model.style_extrap_alpha": 0.2},
    },
    {
        "name": "P2_alpha_030",
        "desc": "style_extrap_alpha=0.30",
        "overrides": {"model.style_extrap_alpha": 0.3},
    },
    # P4: w_endpoint_style sweep (includes L2 at value=0)
    {
        "name": "P4_wstyle_2",
        "desc": "w_endpoint_style=2",
        "overrides": {"bridge.w_endpoint_style": 2},
    },
    {
        "name": "P4_wstyle_4",
        "desc": "w_endpoint_style=4",
        "overrides": {"bridge.w_endpoint_style": 4},
    },
    {
        "name": "P4_wstyle_16",
        "desc": "w_endpoint_style=16",
        "overrides": {"bridge.w_endpoint_style": 16},
    },
    # P5: single_step_swd_weight sweep (includes L4 at value=0)
    {
        "name": "P5_wswd_2",
        "desc": "single_step_swd_weight=2",
        "overrides": {"bridge.single_step_swd_weight": 2},
    },
    {
        "name": "P5_wswd_4",
        "desc": "single_step_swd_weight=4",
        "overrides": {"bridge.single_step_swd_weight": 4},
    },
    {
        "name": "P5_wswd_16",
        "desc": "single_step_swd_weight=16",
        "overrides": {"bridge.single_step_swd_weight": 16},
    },
    # P6: style_cross_attn_gate_init sweep
    {
        "name": "P6_gate_init_0",
        "desc": "style_cross_attn_gate_init=0 (gate starts fully closed)",
        "overrides": {"model.style_cross_attn_gate_init": 0},
    },
    {
        "name": "P6_gate_init_001",
        "desc": "style_cross_attn_gate_init=0.01",
        "overrides": {"model.style_cross_attn_gate_init": 0.01},
    },
    {
        "name": "P6_gate_init_03",
        "desc": "style_cross_attn_gate_init=0.30",
        "overrides": {"model.style_cross_attn_gate_init": 0.3},
    },

    # ===== D. Architecture mode switches (D19-D30, NEW in v2) =====
    {
        "name": "D19_attn_gated_raw",
        "desc": "style_attn_mode=gated_raw (no softmax, raw logits)",
        "overrides": {"model.style_attn_mode": "gated_raw"},
    },
    {
        "name": "D20_attn_relu2",
        "desc": "style_attn_mode=relu2 (relu squared scores)",
        "overrides": {"model.style_attn_mode": "relu2"},
    },
    {
        "name": "D21_attn_style_select",
        "desc": "style_attn_mode=style_select (argmax selection)",
        "overrides": {"model.style_attn_mode": "style_select"},
    },
    {
        "name": "D22_attn_sparsemax",
        "desc": "style_attn_mode=sparsemax (sparse attention)",
        "overrides": {"model.style_attn_mode": "sparsemax"},
    },
    {
        "name": "D23_endpoint_lowhigh",
        "desc": "endpoint_head_mode=endpoint_lowhigh (low/high frequency separation)",
        "overrides": {"model.endpoint_head_mode": "endpoint_lowhigh"},
    },
    {
        "name": "D24_transport_endpoint",
        "desc": "transport_prediction_mode=endpoint (train-side XPred)",
        "overrides": {"model.transport_prediction_mode": "endpoint"},
    },
    {
        "name": "D25_target_proj_dwt",
        "desc": "training_target_projection_mode=dwt (DWT target projection)",
        "overrides": {"bridge.training_target_projection_mode": "dwt"},
    },
    {
        "name": "D26_kinetic_per_band",
        "desc": "kinetic_penalty_mode=per_band (per-band anisotropic kinetic)",
        "overrides": {"bridge.kinetic_penalty_mode": "per_band"},
    },
    {
        "name": "D27_terminal_swd_hf",
        "desc": "terminal_swd_mode=high_freq (high-freq SWD)",
        "overrides": {"bridge.terminal_swd_mode": "high_freq"},
    },
    {
        "name": "D28_bridge_tri_band",
        "desc": "bridge_path_mode=tri_band (tri-band content/style decoupling)",
        "overrides": {"bridge.bridge_path_mode": "tri_band"},
    },
    {
        "name": "D29_swd_squared",
        "desc": "swd_distance_mode=squared (squared SWD distance)",
        "overrides": {"bridge.swd_distance_mode": "squared"},
    },
    {
        "name": "D30_t_logit_normal",
        "desc": "t_sampling_mode=logit_normal (logit-normal time sampling)",
        "overrides": {"bridge.t_sampling_mode": "logit_normal"},
    },

    # ===== E. Loss closure additions (L13-L16, NEW in v2) =====
    {
        "name": "L13_no_flow",
        "desc": "Remove flow matching loss (w_flow=0) - KEY: verify FM dominance",
        "overrides": {"bridge.w_flow": 0},
    },
    {
        "name": "L14_no_coupling_edge",
        "desc": "Remove coupling structure edge weight (coupling_structure_edge_weight=0)",
        "overrides": {"bridge.coupling_structure_edge_weight": 0},
    },
    {
        "name": "L15_no_coupling_hybrid",
        "desc": "Remove coupling structure hybrid stats weight (coupling_structure_hybrid_stats_weight=0)",
        "overrides": {"bridge.coupling_structure_hybrid_stats_weight": 0},
    },
    {
        "name": "L16_no_endpoint_aux",
        "desc": "Remove endpoint aux losses (source_endpoint_aux + endpoint_energy_band)",
        "overrides": {
            "bridge.source_endpoint_aux_weight": 0,
            "bridge.endpoint_energy_band_weight": 0,
        },
    },

    # ===== F. Loss enable exploration (E1-E24, NEW in v2) =====
    # Content fidelity class
    {
        "name": "E1_w_contrast_preserve",
        "desc": "Enable w_contrast_preserve=1.0",
        "overrides": {"bridge.w_contrast_preserve": 1.0},
    },
    {
        "name": "E2_w_channel_variance",
        "desc": "Enable w_channel_variance=1.0",
        "overrides": {"bridge.w_channel_variance": 1.0},
    },
    {
        "name": "E3_w_hf_energy",
        "desc": "Enable w_hf_energy=1.0",
        "overrides": {"bridge.w_hf_energy": 1.0},
    },
    {
        "name": "E4_w_content_lowpass_anchor",
        "desc": "Enable w_content_lowpass_anchor=1.0",
        "overrides": {"bridge.w_content_lowpass_anchor": 1.0},
    },
    {
        "name": "E5_w_content_edge_anchor",
        "desc": "Enable w_content_edge_anchor=1.0",
        "overrides": {"bridge.w_content_edge_anchor": 1.0},
    },
    {
        "name": "E6_w_pixel_color_match",
        "desc": "Enable w_pixel_color_match=1.0",
        "overrides": {"bridge.w_pixel_color_match": 1.0},
    },
    # Style reinforcement class
    {
        "name": "E7_w_velocity_magnitude",
        "desc": "Enable w_velocity_magnitude=1.0",
        "overrides": {"bridge.w_velocity_magnitude": 1.0},
    },
    {
        "name": "E8_w_residual_style_direction",
        "desc": "Enable w_residual_style_direction=1.0",
        "overrides": {"bridge.w_residual_style_direction": 1.0},
    },
    {
        "name": "E9_w_style_contrastive",
        "desc": "Enable w_style_contrastive=1.0",
        "overrides": {"bridge.w_style_contrastive": 1.0},
    },
    {
        "name": "E10_w_style_energy_floor",
        "desc": "Enable w_style_energy_floor=1.0",
        "overrides": {"bridge.w_style_energy_floor": 1.0},
    },
    {
        "name": "E11_w_hsv_saturation",
        "desc": "Enable w_hsv_saturation=1.0",
        "overrides": {"bridge.w_hsv_saturation": 1.0},
    },
    {
        "name": "E12_w_output_variance",
        "desc": "Enable w_output_variance=1.0",
        "overrides": {"bridge.w_output_variance": 1.0},
    },
    # Direction constraint class
    {
        "name": "E13_w_directional_cosine",
        "desc": "Enable w_directional_cosine=1.0",
        "overrides": {"bridge.w_directional_cosine": 1.0},
    },
    {
        "name": "E14_w_freq_split_cosine",
        "desc": "Enable w_freq_split_cosine=1.0 (frequency-band decoupling cosine)",
        "overrides": {"bridge.w_freq_split_cosine": 1.0},
    },
    {
        "name": "E15_w_endpoint_velocity_reg",
        "desc": "Enable w_endpoint_velocity_reg=1.0",
        "overrides": {"bridge.w_endpoint_velocity_reg": 1.0},
    },
    {
        "name": "E16_w_spectral_amplitude",
        "desc": "Enable w_spectral_amplitude=1.0",
        "overrides": {"bridge.w_spectral_amplitude": 1.0},
    },
    # Physics constraint class
    {
        "name": "E17_w_anisotropic_kinetic",
        "desc": "Enable w_anisotropic_kinetic=1.0 (FC-SB fiber bundle theory)",
        "overrides": {"bridge.w_anisotropic_kinetic": 1.0},
    },
    {
        "name": "E18_w_stokes_viscous",
        "desc": "Enable w_stokes_viscous=1.0",
        "overrides": {"bridge.w_stokes_viscous": 1.0},
    },
    {
        "name": "E19_w_curvature",
        "desc": "Enable w_curvature=1.0",
        "overrides": {"bridge.w_curvature": 1.0},
    },
    {
        "name": "E20_w_lowfreq_velocity",
        "desc": "Enable w_lowfreq_velocity=1.0 (FM dominance verification)",
        "overrides": {"bridge.w_lowfreq_velocity": 1.0},
    },
    # Regularization & distillation class
    {
        "name": "E21_w_attn_entropy_reg",
        "desc": "Enable w_attn_entropy_reg=0.5 (Gate Collapse verification)",
        "overrides": {"bridge.w_attn_entropy_reg": 0.5},
    },
    {
        "name": "E22_w_style_strength_reg",
        "desc": "Enable w_style_strength_reg=0.5",
        "overrides": {"bridge.w_style_strength_reg": 0.5},
    },
    {
        "name": "E23_w_variance_penalty",
        "desc": "Enable w_variance_penalty=1.0",
        "overrides": {"bridge.w_variance_penalty": 1.0},
    },
    {
        "name": "E24_w_plain_path_distill",
        "desc": "Enable w_plain_path_distill=1.0",
        "overrides": {"bridge.w_plain_path_distill": 1.0},
    },

    # ===== G. Parameter sweeps extension (P7-P18, NEW in v2) =====
    # P7: spectral_w_hh sweep
    {
        "name": "P7_whh_05",
        "desc": "spectral_w_hh=0.5",
        "overrides": {"bridge.spectral_w_hh": 0.5},
    },
    {
        "name": "P7_whh_30",
        "desc": "spectral_w_hh=3.0",
        "overrides": {"bridge.spectral_w_hh": 3.0},
    },
    {
        "name": "P7_whh_60",
        "desc": "spectral_w_hh=6.0",
        "overrides": {"bridge.spectral_w_hh": 6.0},
    },
    # P8: spectral_w_ll sweep
    {
        "name": "P8_wll_01",
        "desc": "spectral_w_ll=0.1",
        "overrides": {"bridge.spectral_w_ll": 0.1},
    },
    {
        "name": "P8_wll_05",
        "desc": "spectral_w_ll=0.5",
        "overrides": {"bridge.spectral_w_ll": 0.5},
    },
    {
        "name": "P8_wll_20",
        "desc": "spectral_w_ll=2.0",
        "overrides": {"bridge.spectral_w_ll": 2.0},
    },
    # P9: terminal_swd_weight sweep
    {
        "name": "P9_tswd_005",
        "desc": "terminal_swd_weight=0.05",
        "overrides": {"bridge.terminal_swd_weight": 0.05},
    },
    {
        "name": "P9_tswd_05",
        "desc": "terminal_swd_weight=0.5",
        "overrides": {"bridge.terminal_swd_weight": 0.5},
    },
    {
        "name": "P9_tswd_20",
        "desc": "terminal_swd_weight=2.0",
        "overrides": {"bridge.terminal_swd_weight": 2.0},
    },
    # P10: w_kinetic sweep
    {
        "name": "P10_wkin_05",
        "desc": "w_kinetic=0.5",
        "overrides": {"bridge.w_kinetic": 0.5},
    },
    {
        "name": "P10_wkin_20",
        "desc": "w_kinetic=2.0",
        "overrides": {"bridge.w_kinetic": 2.0},
    },
    {
        "name": "P10_wkin_40",
        "desc": "w_kinetic=4.0",
        "overrides": {"bridge.w_kinetic": 4.0},
    },
    {
        "name": "P10_wkin_80",
        "desc": "w_kinetic=8.0",
        "overrides": {"bridge.w_kinetic": 8.0},
    },
    # P11: bridge_sigma sweep
    {
        "name": "P11_sigma_000",
        "desc": "bridge_sigma=0.0",
        "overrides": {"bridge.bridge_sigma": 0.0},
    },
    {
        "name": "P11_sigma_005",
        "desc": "bridge_sigma=0.05",
        "overrides": {"bridge.bridge_sigma": 0.05},
    },
    {
        "name": "P11_sigma_008",
        "desc": "bridge_sigma=0.08 (magic threshold)",
        "overrides": {"bridge.bridge_sigma": 0.08},
    },
    {
        "name": "P11_sigma_010",
        "desc": "bridge_sigma=0.1",
        "overrides": {"bridge.bridge_sigma": 0.1},
    },
    # P12: single_step_edge_weight sweep
    {
        "name": "P12_edge_005",
        "desc": "single_step_edge_weight=0.05",
        "overrides": {"bridge.single_step_edge_weight": 0.05},
    },
    {
        "name": "P12_edge_05",
        "desc": "single_step_edge_weight=0.5",
        "overrides": {"bridge.single_step_edge_weight": 0.5},
    },
    {
        "name": "P12_edge_10",
        "desc": "single_step_edge_weight=1.0",
        "overrides": {"bridge.single_step_edge_weight": 1.0},
    },
    {
        "name": "P12_edge_20",
        "desc": "single_step_edge_weight=2.0",
        "overrides": {"bridge.single_step_edge_weight": 2.0},
    },
    # P13: w_flow sweep - KEY
    {
        "name": "P13_wflow_01",
        "desc": "w_flow=0.1 (reduce FM dominance)",
        "overrides": {"bridge.w_flow": 0.1},
    },
    {
        "name": "P13_wflow_03",
        "desc": "w_flow=0.3",
        "overrides": {"bridge.w_flow": 0.3},
    },
    {
        "name": "P13_wflow_05",
        "desc": "w_flow=0.5",
        "overrides": {"bridge.w_flow": 0.5},
    },
    {
        "name": "P13_wflow_20",
        "desc": "w_flow=2.0",
        "overrides": {"bridge.w_flow": 2.0},
    },
    # P14: w_endpoint_content sweep
    {
        "name": "P14_wcontent_05",
        "desc": "w_endpoint_content=0.5",
        "overrides": {"bridge.w_endpoint_content": 0.5},
    },
    {
        "name": "P14_wcontent_20",
        "desc": "w_endpoint_content=2.0",
        "overrides": {"bridge.w_endpoint_content": 2.0},
    },
    {
        "name": "P14_wcontent_40",
        "desc": "w_endpoint_content=4.0",
        "overrides": {"bridge.w_endpoint_content": 4.0},
    },
    {
        "name": "P14_wcontent_80",
        "desc": "w_endpoint_content=8.0",
        "overrides": {"bridge.w_endpoint_content": 8.0},
    },
    # P15: coupling_structure_cost_weight sweep
    {
        "name": "P15_coupling_05",
        "desc": "coupling_structure_cost_weight=0.5",
        "overrides": {"bridge.coupling_structure_cost_weight": 0.5},
    },
    {
        "name": "P15_coupling_20",
        "desc": "coupling_structure_cost_weight=2.0",
        "overrides": {"bridge.coupling_structure_cost_weight": 2.0},
    },
    {
        "name": "P15_coupling_40",
        "desc": "coupling_structure_cost_weight=4.0",
        "overrides": {"bridge.coupling_structure_cost_weight": 4.0},
    },
    {
        "name": "P15_coupling_80",
        "desc": "coupling_structure_cost_weight=8.0",
        "overrides": {"bridge.coupling_structure_cost_weight": 8.0},
    },
    # P16: style_attn_num_tokens sweep
    {
        "name": "P16_tokens_64",
        "desc": "style_attn_num_tokens=64",
        "overrides": {"model.style_attn_num_tokens": 64},
    },
    {
        "name": "P16_tokens_128",
        "desc": "style_attn_num_tokens=128",
        "overrides": {"model.style_attn_num_tokens": 128},
    },
    {
        "name": "P16_tokens_512",
        "desc": "style_attn_num_tokens=512",
        "overrides": {"model.style_attn_num_tokens": 512},
    },
    {
        "name": "P16_tokens_1024",
        "desc": "style_attn_num_tokens=1024",
        "overrides": {"model.style_attn_num_tokens": 1024},
    },
    # P17: style_attn_sharpen_scale sweep
    {
        "name": "P17_sharpen_25",
        "desc": "style_attn_sharpen_scale=2.5",
        "overrides": {"model.style_attn_sharpen_scale": 2.5},
    },
    {
        "name": "P17_sharpen_50",
        "desc": "style_attn_sharpen_scale=5.0",
        "overrides": {"model.style_attn_sharpen_scale": 5.0},
    },
    {
        "name": "P17_sharpen_100",
        "desc": "style_attn_sharpen_scale=10.0",
        "overrides": {"model.style_attn_sharpen_scale": 10.0},
    },
    # P18: style_cross_attn_gate_init extension (large gate values)
    {
        "name": "P18_gate_init_01",
        "desc": "style_cross_attn_gate_init=0.1",
        "overrides": {"model.style_cross_attn_gate_init": 0.1},
    },
    {
        "name": "P18_gate_init_05",
        "desc": "style_cross_attn_gate_init=0.5",
        "overrides": {"model.style_cross_attn_gate_init": 0.5},
    },
    {
        "name": "P18_gate_init_10",
        "desc": "style_cross_attn_gate_init=1.0 (fully open gate)",
        "overrides": {"model.style_cross_attn_gate_init": 1.0},
    },
]


def generate_configs():
    with open(T5_CONFIG_PATH, "r", encoding="utf-8") as f:
        base = json.load(f)

    is_win = sys.platform == "win32"

    for abl in ABLATIONS:
        cfg = json.loads(json.dumps(base))

        cfg["checkpoint"]["save_dir"] = f"./exp/628_ablation/destructive/{abl['name']}"
        cfg["training"]["num_epochs"] = 10
        cfg["training"]["save_interval"] = 1
        cfg["training"]["full_eval_each_epoch"] = True
        cfg["training"]["full_eval_defer_until_training_end"] = False
        cfg["training"]["resume_training_state"] = True
        cfg["training"]["resume_optimizer"] = True
        cfg["training"]["resume_model_strict"] = abl.get("strict_resume", True)

        cfg["ablation"] = {
            "name": abl["name"],
            "axis": "628_destructive",
            "stage": "ep8-10",
            "notes": abl["desc"],
        }

        for dotted_key, value in abl["overrides"].items():
            parts = dotted_key.split(".")
            section = parts[0]
            key = parts[1]
            cfg.setdefault(section, {})[key] = value

        if is_win:
            path_fixes = {
                "data.data_root": "I:/wikiart_distinct5_samam_512_latents_ema/train",
                "training.test_image_dir": "I:/wikiart_distinct5_samam_512_classview/test",
                "training.full_eval_cache_dir": "I:/Github/Latent_Style/eval_cache",
                "training.full_eval_clip_hf_cache_dir": "I:/Github/Latent_Style/eval_cache/hf",
                "data.latent_cache_dir": "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed",
                "data.pairing_cache_path": "I:/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt",
            }
        else:
            path_fixes = {
                "data.data_root": "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train",
                "training.test_image_dir": "/mnt/i/wikiart_distinct5_samam_512_classview/test",
                "training.full_eval_cache_dir": "/mnt/i/eval_cache",
                "training.full_eval_clip_hf_cache_dir": "/mnt/i/eval_cache/hf",
                "data.latent_cache_dir": "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed",
                "data.pairing_cache_path": "/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt",
            }
        for dotted_key, value in path_fixes.items():
            parts = dotted_key.split(".")
            cfg[parts[0]][parts[1]] = value

        if is_win:
            cfg["training"]["resume_checkpoint"] = "I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/t5_b2v2_d2_d4/epoch_0007.pt"
        else:
            cfg["training"]["resume_checkpoint"] = "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/t5_b2v2_d2_d4/epoch_0007.pt"

        out_path = OUTPUT_DIR / f"{abl['name']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"  Generated {out_path.name}: {abl['desc']}")


if __name__ == "__main__":
    print(f"Generating {len(ABLATIONS)} destructive ablation configs from T5 baseline...")
    generate_configs()
    print(f"Done. Output: {OUTPUT_DIR}")
