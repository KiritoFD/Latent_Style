"""Generate Phase 0-2 OMF experiments with mathematical reasoning"""
import json, os

ROOT = "G:/GitHub/Latent_Style/SchrodingerBridge"

# Base: D0 config (known working: 0.701 style with K1)
d0_path = os.path.join(ROOT, "ablation_destructive_7epoch", "configs", "D0_full_correct_7ep.json")
with open(d0_path) as f:
    base = json.load(f)

# Remove the ablation metadata to keep config clean
base.pop("ablation", None)

EXPERIMENTS = [
    # === Phase 0: Baseline Reproduction ===
    # Theory: D0 config with correct params. terminal_swd_weight=20 (vs 0.15 in our failed OMF-0).
    # The SWD term goes from 2.3% of loss to ~98% of loss. Model must optimize style, not low-freq structure.
    ("K1_W20", {
        "w_kinetic": 1.0, "terminal_swd_weight": 20.0,
        "w_low_freq": 0.0, "w_cycle": 0.0, "w_color": 0.0,
        "kinetic_entropy_gate_weight": 0.0,
        "note": "Phase 0: Reproduce D0 baseline (expect 0.700-0.705 style)"
    }, 12),

    # === Phase 1: Low kinetic + entropy gate ===
    # Theory: Reduce kinetic weight 1.0→0.5 to allow more velocity (more style).
    # Enable entropy gate (weight=2.0) to protect uncertain regions from content drift.
    # Expected: style +0.008~0.012, LPIPS similar or slightly improved.
    ("K0p5_W20_ent2", {
        "w_kinetic": 0.5, "terminal_swd_weight": 20.0,
        "w_low_freq": 0.0, "w_cycle": 0.0, "w_color": 0.0,
        "kinetic_entropy_gate_weight": 2.0,
        "note": "Phase 1: kinetic=0.5 + entropy gate=2.0. More style, adaptive content protection."
    }, 12),
    
    # === Phase 2a: Aggressive style push ===
    # Theory: kinetic=0.25 (+entropy=3) + SWD=30.
    # Ratio w_swd/w_kin = 30/0.25 = 120 (vs 20 in K1 baseline). 6x more aggressive style signal.
    # Entropy gate (weight=3) provides 4x adaptive kinetic boost in uncertain regions.
    # Expected: style ~0.718-0.725
    ("K0p25_W30_ent3", {
        "w_kinetic": 0.25, "terminal_swd_weight": 30.0,
        "w_low_freq": 0.0, "w_cycle": 0.0, "w_color": 0.0,
        "kinetic_entropy_gate_weight": 3.0,
        "note": "Phase 2a: kinetic=0.25 + SWD=30 + entropy=3. Target 0.72+ style."
    }, 12),
    
    # === Phase 2b: Max SWD ===
    # Theory: Same as 2a but SWD=40. Ratio = 40/0.25 = 160 (8x baseline).
    # Style ceiling exploration. Risk: LPIPS may be too high.
    ("K0p25_W40_ent3", {
        "w_kinetic": 0.25, "terminal_swd_weight": 40.0,
        "w_low_freq": 0.0, "w_cycle": 0.0, "w_color": 0.0,
        "kinetic_entropy_gate_weight": 3.0,
        "note": "Phase 2b: kinetic=0.25 + SWD=40 + entropy=3. Max style push."
    }, 12),
    
    # === Phase 2c: Style ceiling (no kinetic) ===
    # Theory: w_kinetic=0 means velocity is completely unconstrained.
    # This gives the upper style bound. D2 got 0.716 style at W=20.
    # With W=30, estimate ~0.720+. But LPIPS will be high (~0.6+).
    ("K0_W30", {
        "w_kinetic": 0.0, "terminal_swd_weight": 30.0,
        "w_low_freq": 0.0, "w_cycle": 0.0, "w_color": 0.0,
        "kinetic_entropy_gate_weight": 0.0,
        "note": "Phase 2c: No kinetic. Upper style bound at W=30. LPIPS warning."
    }, 12),
    
    # === Speed test ===
    # Theory: Reduced SWD patches [3,5] (vs [3,5,7,15]) and projections 32 (vs 64).
    # SWD cost scales O(patch_sizes × projections). Reduction: 2/4 × 32/64 = 25% of original.
    # Expected speedup: ~30% per epoch. Quality impact: TBD.
    ("K0p5_W20_ent2_fast", {
        "w_kinetic": 0.5, "terminal_swd_weight": 20.0,
        "w_low_freq": 0.0, "w_cycle": 0.0, "w_color": 0.0,
        "kinetic_entropy_gate_weight": 2.0,
        "swd_num_projections": 32,
        "semantic_swd_num_projections": 32,
        "note": "Speed test: reduced patches+projections vs entropy baseline."
    }, 12),
]

os.makedirs(os.path.join(ROOT, "omf_sweep3"), exist_ok=True)

for name, overrides, num_epochs in EXPERIMENTS:
    exp_dir = os.path.join(ROOT, "omf_sweep3", name)
    os.makedirs(exp_dir, exist_ok=True)
    cfg = json.loads(json.dumps(base))
    
    # Apply bridge overrides
    cfg["bridge"].update(overrides)
    
    # Set model params (D0 correct values that differ from our wrong config)
    cfg["model"]["semantic_attn_temperature"] = 0.12
    cfg["model"]["skip_routing_mode"] = "add_proj"
    cfg["model"]["skip_fusion_mode"] = "add_proj"
    
    # Training params
    cfg["training"]["num_epochs"] = num_epochs
    cfg["training"]["save_interval"] = 1
    cfg["training"]["batch_size"] = 32
    cfg["training"]["full_eval_batch_size"] = 6
    cfg["checkpoint"]["save_dir"] = f"./omf_sweep3/{name}/checkpoints"
    
    path = os.path.join(exp_dir, "config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    
    b, t = cfg["bridge"], cfg["training"]
    eg = b.get("kinetic_entropy_gate_weight", 0)
    print(f"{name:25s} K={b['w_kinetic']:<5} W={b['terminal_swd_weight']:<5} "
          f"ent={eg:<4} ratio={b['terminal_swd_weight']/max(b['w_kinetic'],1e-6):.0f} "
          f"ep={t['num_epochs']:<3} bs={t['batch_size']:<3} | {b.get('note','')}")
