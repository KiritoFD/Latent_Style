"""Generate comprehensive 512 ablation configs (v3, 48 configs across 9 axes).

Design principles (per user feedback "更全面、配置更极端"):
- 9 theoretical axes (solver, spectral, adain, bridge, coupling, loss, style, arch, training)
- 48 configs total (vs previous 19), each axis has 4-7 configs
- EXTREME values: 0.0, 4.0, 32.0, 64.0, num_steps=1/32, levels=5, etc.
- Each config: clear hypothesis + predicted effect + axis tag
- All inherit from 630_phase4i2b_sota_heun_5ep.json (current SOTA)
- All use remote I drive paths
- batch_size=16 (match SOTA exactly for fair comparison; 24 may OOM at 512)

Usage:
    python scripts/_gen_abl512_v3.py
    # generates configs/abl512_X*.json
"""
import json
import os
from pathlib import Path

# === Remote paths (I drive, Windows format for Windows python) ===
# Datasets unified under I:/datasets/ (moved 2026-07-06)
R_REPO = "I:/Github/Latent_Style/SchrodingerBridge"
R_DATA = "I:/datasets/wikiart_distinct5_samam_512_latents_ema/train"
R_TEST = "I:/datasets/wikiart_distinct5_samam_512_classview/test"
R_CACHE = "I:/datasets/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed"
R_PAIR = "I:/datasets/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt"
R_EXP = f"{R_REPO}/exp/abl512"
R_EVAL_CACHE = f"{R_REPO}/exp/eval_cache"
R_HF_CACHE = f"{R_REPO}/exp/eval_cache/hf"

BASE_CONFIG = "630_phase4i2b_sota_heun_5ep.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "configs"

# Common remote overrides (paths + cache + eval batch)
COMMON = {
    "data": {
        "data_root": R_DATA,
        "latent_cache_dir": R_CACHE,
        "pairing_cache_path": R_PAIR,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    },
    "training": {
        "batch_size": 16,  # match SOTA exactly for fair comparison
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "test_image_dir": R_TEST,
        "full_eval_cache_dir": R_EVAL_CACHE,
        "full_eval_clip_hf_cache_dir": R_HF_CACHE,
        "num_epochs": 5,
        "patience": 2,
        "full_eval_each_epoch": True,
    },
    "full_eval": {
        "batch_size": 2,
        "ref_feature_batch_size": 2,
        "vae_model": "ema",
        "num_steps": 8,
    },
}


def make_config(name: str, axis: str, notes: str, model=None, bridge=None,
                training=None, full_eval=None, data=None) -> dict:
    """Build a config dict inheriting from SOTA + common remote overrides + ablation."""
    cfg = {
        "_base": BASE_CONFIG,
        "checkpoint": {
            "save_dir": f"{R_EXP}/{name}",
            "resume_checkpoint": "",
        },
        "ablation": {
            "name": name,
            "axis": axis,
            "stage": "abl512_v3",
            "notes": notes,
        },
    }
    # Apply common overrides
    for section, vals in COMMON.items():
        cfg.setdefault(section, {})
        cfg[section].update(vals)
    # Apply ablation-specific overrides
    if model:
        cfg.setdefault("model", {})
        cfg["model"].update(model)
    if bridge:
        cfg.setdefault("bridge", {})
        cfg["bridge"].update(bridge)
    if training:
        cfg.setdefault("training", {})
        cfg["training"].update(training)
    if full_eval:
        cfg.setdefault("full_eval", {})
        cfg["full_eval"].update(full_eval)
    if data:
        cfg.setdefault("data", {})
        cfg["data"].update(data)
    return cfg


# ====================================================================
# AXIS A: ODE Solver (5 configs) — tests solver order, steps, corrector
# ====================================================================
AXIS_A = [
    ("X01_euler", "solver_order",
     "Downgrade solver: heun->euler. Euler O(h^2) vs Heun O(h^3). Predict: -0.005 CLIP-S, +0.01 LPIPS.",
     {"model": {"solver_type": "euler"}}),
    ("X02_rk4", "solver_order",
     "Upgrade solver: heun->rk4 (4th order). Extreme high-order. Predict: marginal gain (saturates, see 4I.2b notes).",
     {"model": {"solver_type": "rk4"}}),
    ("X03_steps_1", "inference_steps",
     "Extreme: 1-step inference. Tests if model can do single-step style transfer. Predict: -0.02 CLIP-S, big LPIPS degradation.",
     {"full_eval": {"num_steps": 1}}),
    ("X04_steps_32", "inference_steps",
     "Extreme: 32-step inference. Diminishing returns test. Predict: +0.001 CLIP-S vs 8-step, 4x slower.",
     {"full_eval": {"num_steps": 32}}),
    ("X05_corrector_4", "solver_corrector",
     "Heavy corrector: 4 corrector steps. Predict: marginal quality gain, 4x slower per step.",
     {"model": {"solver_corrector_steps": 4, "solver_corrector_mode": "none"}}),
]

# ====================================================================
# AXIS B: Spectral ODE / Frequency (6 configs) — tests DWT core
# ====================================================================
AXIS_B = [
    ("X06_no_spectral_ode", "spectral_core",
     "DESTRUCTIVE: disable spectral_ode entirely. Predict: large degradation, validates DWT necessity.",
     {"model": {"spectral_ode_enabled": False}}),
    ("X07_spectral_levels_4", "spectral_depth",
     "Extreme deep: 4-level DWT (vs SOTA 1). LL_4 (2x2) near-global. Predict: over-decomposition, structural loss.",
     {"model": {"spectral_ode_levels": 4}}),
    ("X08_spectral_levels_5", "spectral_depth",
     "EXTREME: 5-level DWT. LL_5 (1x1) = scalar. Predict: catastrophic, validates 3-level peak.",
     {"model": {"spectral_ode_levels": 5}}),
    ("X09_lowpass_avg", "lowpass_mode",
     "Replace DWT with avg_pool. Loses frequency localization. Predict: -0.01 CLIP-S.",
     {"model": {"lowpass_mode": "avg_pool"}}),
    ("X10_w_ll_0", "spectral_weights",
     "Zero LL weight: ignore low-freq. Predict: content collapse, LPIPS explosion.",
     {"bridge": {"spectral_w_ll": 0.0}}),
    ("X11_w_hh_3x", "spectral_weights",
     "3x HH weight: extreme high-freq emphasis. Predict: texture over-emphasis, content loss.",
     {"bridge": {"spectral_w_hl": 3.0, "spectral_w_lh": 3.0}}),
]

# ====================================================================
# AXIS C: AdaIN / Endpoint (7 configs) — tests style injection
# ====================================================================
AXIS_C = [
    ("X12_adain_0", "adain_scale",
     "DESTRUCTIVE: AdaIN scale=0. No style injection at endpoint. Predict: clip near identity (0.6933).",
     {"model": {"endpoint_adain_scale": 0.0}}),
    ("X13_adain_4x", "adain_scale",
     "EXTREME: AdaIN scale=4.0 (5x SOTA). Predict: style explosion, content collapse.",
     {"model": {"endpoint_adain_scale": 4.0}}),
    ("X14_adain_every_step", "adain_schedule",
     "Apply AdaIN every step (vs only_last_step=True). Predict: alpha invalidated (4G.2b lesson).",
     {"model": {"endpoint_adain_only_last_step": False}}),
    ("X15_lowpass_1", "lowpass_depth",
     "Shallow: 1-level lowpass (vs 3). Predict: less structural protection, +0.005 LPIPS.",
     {"model": {"endpoint_lowpass_levels": 1}}),
    ("X16_lowpass_5", "lowpass_depth",
     "EXTREME deep: 5-level lowpass. Predict: over-smoothing, structural blur.",
     {"model": {"endpoint_lowpass_levels": 5}}),
    ("X17_velocity_floor_0", "velocity_floor",
     "Remove velocity floor. Predict: small velocity at endpoints causes noise.",
     {"model": {"endpoint_velocity_floor": 0.0}}),
    ("X18_velocity_floor_0p3", "velocity_floor",
     "EXTREME high velocity floor=0.3. Predict: forced large velocity, instability.",
     {"model": {"endpoint_velocity_floor": 0.3}}),
]

# ====================================================================
# AXIS D: Bridge Path (5 configs) — tests trajectory
# ====================================================================
AXIS_D = [
    ("X19_path_linear", "bridge_path",
     "Linear path (vs tri_band). Predict: loses frequency-aware interpolation, -0.005 CLIP-S.",
     {"bridge": {"bridge_path_mode": "linear"}}),
    ("X20_path_slerp", "bridge_path",
     "Spherical linear interpolation. Predict: norm-preserving but may mismatch latent geometry.",
     {"bridge": {"bridge_path_mode": "slerp"}}),
    ("X21_sigma_0", "bridge_noise",
     "DESTRUCTIVE: zero bridge noise. Deterministic path. Predict: overfitting, reduced diversity.",
     {"bridge": {"bridge_sigma": 0.0}}),
    ("X22_sigma_0p5", "bridge_noise",
     "EXTREME: 25x bridge noise. Predict: trajectory collapses, near-pure-noise.",
     {"bridge": {"bridge_sigma": 0.5}}),
    ("X23_no_target_proj", "target_projection",
     "Disable DWT target projection. Predict: high-freq leakage into training target.",
     {"bridge": {"training_target_projection_mode": "none"}}),
]

# ====================================================================
# AXIS E: Coupling / OT (5 configs) — tests optimal transport
# ====================================================================
AXIS_E = [
    ("X24_hungarian", "coupling_solver",
     "Optimal OT: Hungarian (vs independent). Predict: better pairings, slower, marginal CLIP-S gain.",
     {"bridge": {"coupling_solver": "hungarian"}}),
    ("X25_no_structure_cost", "structure_cost",
     "DESTRUCTIVE: zero structure cost. Predict: random pairings, large LPIPS increase.",
     {"bridge": {"coupling_structure_cost_weight": 0.0}}),
    ("X26_structure_5x", "structure_cost",
     "5x structure cost. Predict: overly conservative pairings, reduced style diversity.",
     {"bridge": {"coupling_structure_cost_weight": 5.0}}),
    ("X27_sinkhorn_eps_0p5", "sinkhorn_eps",
     "EXTREME blurry OT: epsilon=0.5. Predict: near-uniform coupling, style mixing.",
     {"bridge": {"sinkhorn_epsilon": 0.5}}),
    ("X28_sinkhorn_iters_10", "sinkhorn_iters",
     "Low-quality OT: 10 iters (vs 60). Predict: under-converged coupling, minor degradation.",
     {"bridge": {"sinkhorn_iters": 10}}),
]

# ====================================================================
# AXIS F: Loss Weights (7 configs) — tests objective balance
# ====================================================================
AXIS_F = [
    ("X29_no_content_loss", "w_content",
     "DESTRUCTIVE: zero content loss. Predict: content collapse, LPIPS explosion.",
     {"bridge": {"w_endpoint_content": 0.0}}),
    ("X30_content_5x", "w_content",
     "5x content weight. Predict: content-preserved but style-weak, CLIP-S drops.",
     {"bridge": {"w_endpoint_content": 5.0}}),
    ("X31_no_style_loss", "w_style",
     "DESTRUCTIVE: zero style loss. Predict: no style transfer, clip=identity.",
     {"bridge": {"w_endpoint_style": 0.0}}),
    ("X32_style_32x", "w_style",
     "EXTREME: 32x style weight (4x SOTA). Predict: style explosion, content collapse.",
     {"bridge": {"w_endpoint_style": 32.0}}),
    ("X33_style_64x", "w_style",
     "EXTREME: 64x style weight (8x SOTA). Predict: catastrophic content loss.",
     {"bridge": {"w_endpoint_style": 64.0}}),
    ("X34_no_flow", "w_flow",
     "DESTRUCTIVE: zero flow loss. Predict: model cannot learn trajectory, NaN/instability.",
     {"bridge": {"w_flow": 0.0}}),
    ("X35_no_kinetic", "w_kinetic",
     "Zero kinetic loss. Predict: less smooth trajectories, marginal effect.",
     {"bridge": {"w_kinetic": 0.0}}),
]

# ====================================================================
# AXIS G: Style / Tokenizer (5 configs) — tests style encoder
# ====================================================================
AXIS_G = [
    ("X36_attn_softmax", "attn_mode",
     "Softmax attention (vs relu2). Historical bug comparison (4I.2a fixed relu2). Predict: -0.005.",
     {"model": {"style_attn_mode": "softmax"}}),
    ("X37_heads_1", "attn_heads",
     "Minimal: 1 attention head. Predict: reduced style capacity, -0.005 CLIP-S.",
     {"model": {"style_attn_num_heads": 1}}),
    ("X38_heads_16", "attn_heads",
     "EXTREME: 16 attention heads (4x SOTA). Predict: over-parameterized, possible overfit.",
     {"model": {"style_attn_num_heads": 16}}),
    ("X39_no_shortcut", "style_shortcut",
     "Zero style shortcut. Predict: loses residual style path, -0.01 CLIP-S.",
     {"model": {"style_shortcut_alpha": 0.0}}),
    ("X40_extrap_1", "style_extrap",
     "EXTREME: extrap_alpha=1.0 (10x SOTA). Predict: extrapolation instability.",
     {"model": {"style_extrap_alpha": 1.0}}),
]

# ====================================================================
# AXIS H: Architecture (4 configs) — tests capacity
# ====================================================================
AXIS_H = [
    ("X41_dim_32", "model_capacity",
     "Half capacity: base_dim=32. Predict: under-fit, -0.01 both metrics.",
     {"model": {"base_dim": 32}}),
    ("X42_dim_128", "model_capacity",
     "2x capacity: base_dim=128. Predict: over-parameterized, may overfit, slower.",
     {"model": {"base_dim": 128}}),
    ("X43_res_blocks_2", "depth",
     "Shallow: 2 res blocks (vs 4). Predict: reduced capacity, -0.005 CLIP-S.",
     {"model": {"num_res_blocks": 2}}),
    ("X44_no_skip", "skip_fusion",
     "No skip fusion. Predict: loses multi-scale fusion, content loss.",
     {"model": {"skip_fusion_mode": "none"}}),
]

# ====================================================================
# AXIS I: Training Schedule (4 configs) — tests optimization
# ====================================================================
AXIS_I = [
    ("X45_epochs_1", "training_length",
     "Minimal: 1 epoch. Predict: under-trained, large degradation.",
     {"training": {"num_epochs": 1, "patience": 1}}),
    ("X46_lr_10x", "learning_rate",
     "EXTREME: 10x learning rate. Predict: divergence/instability.",
     {"training": {"learning_rate": 0.002}}),
    ("X47_lr_0p1x", "learning_rate",
     "EXTREME: 1/10 learning rate. Predict: under-trained, slow convergence.",
     {"training": {"learning_rate": 0.00002}}),
    ("X48_t_uniform", "t_sampling",
     "Uniform t sampling (vs logit_normal). Predict: less focus on mid-trajectory, marginal.",
     {"bridge": {"t_sampling_mode": "uniform"}}),
]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    all_configs = (
        [(c, "A_solver") for c in AXIS_A] +
        [(c, "B_spectral") for c in AXIS_B] +
        [(c, "C_adain") for c in AXIS_C] +
        [(c, "D_bridge") for c in AXIS_D] +
        [(c, "E_coupling") for c in AXIS_E] +
        [(c, "F_loss") for c in AXIS_F] +
        [(c, "G_style") for c in AXIS_G] +
        [(c, "H_arch") for c in AXIS_H] +
        [(c, "I_training") for c in AXIS_I]
    )
    print(f"=== Generating {len(all_configs)} ablation configs ===\n")
    print(f"{'#':>3}  {'Name':30s}  {'Axis':18s}  File")
    print("-" * 90)
    for i, ((name, axis, notes, overrides), axis_group) in enumerate(all_configs, 1):
        # Unpack overrides
        cfg = make_config(name, axis, notes,
                          model=overrides.get("model"),
                          bridge=overrides.get("bridge"),
                          training=overrides.get("training"),
                          full_eval=overrides.get("full_eval"),
                          data=overrides.get("data"))
        out_path = OUT_DIR / f"abl512_{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"{i:>3}  {name:30s}  {axis_group:18s}  {out_path.name}")
    print(f"\n=== Done. {len(all_configs)} configs in {OUT_DIR} ===")
    # Write a manifest for the batch runner
    manifest = []
    for (name, axis, notes, _), axis_group in all_configs:
        manifest.append({"name": name, "axis": axis, "axis_group": axis_group, "notes": notes})
    manifest_path = OUT_DIR / "_abl512_v3_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
