"""Collect ablation results from remote experiments."""
import json, os, sys, glob

# Experiments to collect
EXPERIMENTS = [
    # User-provided (already run as abl_min_*)
    ("abl_min_attention_off", "Cross-attention off"),
    ("abl_min_ll0", "LL weight = 0"),
    ("abl_min_ll1", "LL weight = 1.0"),
    ("abl_min_sigma0", "Sigma = 0"),
    # Clean baseline
    ("refactor_clean_baseline", "Baseline (clean)"),
    # Previous destructive ablations
    ("wo_endpoint_adain", "wo_endpoint_adain"),
    ("wo_flow", "wo_flow"),
    ("wo_cross_attn", "wo_cross_attn"),
    ("wo_spectral_ode", "wo_spectral_ode"),
    ("wo_wavelet", "wo_wavelet"),
    ("wo_asg", "wo_asg"),
    # New full ablation (if exist)
    ("ablation_full/wll_01", "wll_01"),
    ("ablation_full/wll_10", "wll_10"),
    ("ablation_full/wes_02", "wes_02"),
    ("ablation_full/wes_16", "wes_16"),
    ("ablation_full/sigma_005", "sigma_005"),
    ("ablation_full/gate_03", "gate_03"),
    ("ablation_full/adain_05", "adain_05"),
    ("ablation_full/adain_20", "adain_20"),
]

BASE = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
DINO_BASE = r"I:\Github\Latent_Style\SchrodingerBridge\state\dino"

results = []
for exp_dir, name in EXPERIMENTS:
    full_path = os.path.join(BASE, exp_dir)
    # Try multiple summary locations
    summary_paths = [
        os.path.join(full_path, "eval", "summary.json"),
        os.path.join(full_path, "full_eval", "epoch_0005", "summary.json"),
        os.path.join(full_path, "summary.json"),
    ]
    summary = None
    for sp in summary_paths:
        if os.path.exists(sp):
            with open(sp) as f:
                summary = json.load(f)
            break

    # DINO results
    dino_paths = [
        os.path.join(full_path, "eval", "dino_summary.json"),
        os.path.join(DINO_BASE, f"D5-512__{os.path.basename(exp_dir)}.json"),
    ]
    dino = None
    for dp in dino_paths:
        if os.path.exists(dp):
            with open(dp) as f:
                dino = json.load(f)
            break

    row = {"name": name, "exp_dir": exp_dir}
    if summary:
        # Extract aggregate metrics
        agg = summary.get("aggregate", summary.get("summary", {}))
        if "clip_s" in agg:
            row["clip_s"] = agg["clip_s"].get("mean") if isinstance(agg["clip_s"], dict) else agg["clip_s"]
        elif "all_clip_s" in summary:
            row["clip_s"] = summary["all_clip_s"]
        if "lpips" in agg:
            row["lpips"] = agg["lpips"].get("mean") if isinstance(agg["lpips"], dict) else agg["lpips"]
        elif "all_lpips" in summary:
            row["lpips"] = summary["all_lpips"]
    if dino:
        if "all_dino_s" in dino:
            row["dino_s"] = dino["all_dino_s"]
            row["dino_c"] = dino["all_dino_c"]
        elif "dino_style" in dino:
            row["dino_s"] = dino["dino_style"]
            row["dino_c"] = dino["dino_content"]
        # DINO summary also contains CLIP-S and LPIPS
        if "all_clip_s" in dino and "clip_s" not in row:
            row["clip_s"] = dino["all_clip_s"]
        if "all_lpips" in dino and "lpips" not in row:
            row["lpips"] = dino["all_lpips"]

    results.append(row)

# Print as table
print(f"{'Name':<30} {'CLIP-S':>8} {'LPIPS':>8} {'DINO-S':>8} {'DINO-C':>8}")
print("-" * 70)
for r in results:
    clip_s = f"{r.get('clip_s', 0):.4f}" if r.get('clip_s') is not None else "  —   "
    lpips = f"{r.get('lpips', 0):.4f}" if r.get('lpips') is not None else "  —   "
    dino_s = f"{r.get('dino_s', 0):.4f}" if r.get('dino_s') is not None else "  —   "
    dino_c = f"{r.get('dino_c', 0):.4f}" if r.get('dino_c') is not None else "  —   "
    print(f"{r['name']:<30} {clip_s:>8} {lpips:>8} {dino_s:>8} {dino_c:>8}")

# Also dump as JSON
print("\n--- JSON ---")
print(json.dumps(results, indent=2, default=str))
