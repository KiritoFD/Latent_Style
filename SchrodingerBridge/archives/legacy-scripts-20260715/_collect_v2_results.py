"""Collect all ablation_v2 results — final version."""
import json, os

BASE = r"I:\Github\Latent_Style\SchrodingerBridge\exp\ablation_v2"

EXPS = [
    "a01_wo_endpoint_adain", "a02_wo_cross_attn", "a03_wo_flow",
    "b01_wll_0", "b02_wll_20",
    "b03_sigma_0", "b04_sigma_02",
    "b05_gate_001", "b06_gate_10",
    "b07_whh_0", "b08_whh_4",
    "b09_lr_5e5", "b10_lr_5e4",
    "b11_loss_huber",
    "d01_adain_0", "d02_adain_05", "d03_adain_20",
    "d04_extrap_00", "d05_extrap_10",
    "d06_steps_1", "d07_steps_32",
]

results = []
for name in EXPS:
    exp_dir = os.path.join(BASE, name)
    clip_s = lpips = dino_s = dino_c = None

    # Training experiments: summary in eval/summary.json
    # Inference experiments: summary in summary.json (root)
    summary_paths = [
        os.path.join(exp_dir, "eval", "summary.json"),
        os.path.join(exp_dir, "summary.json"),
    ]
    for sp in summary_paths:
        if os.path.exists(sp):
            with open(sp) as f:
                d = json.load(f)
            overview = d.get("analysis", {}).get("all_pairs_overview", {})
            clip_s = overview.get("clip_style")
            lpips = overview.get("content_lpips")
            break

    # DINO: training in eval/dino_summary.json, inference in dino_summary.json
    dino_paths = [
        os.path.join(exp_dir, "eval", "dino_summary.json"),
        os.path.join(exp_dir, "dino_summary.json"),
    ]
    for dp in dino_paths:
        if os.path.exists(dp):
            with open(dp) as f:
                d = json.load(f)
            dino_s = d.get("all_dino_s", d.get("dino_style"))
            dino_c = d.get("all_dino_c", d.get("dino_content"))
            break

    results.append({"name": name, "clip_s": clip_s, "lpips": lpips, "dino_s": dino_s, "dino_c": dino_c})

# Print table
print(f"{'Name':<25} {'CLIP-S':>8} {'LPIPS':>8} {'DINO-S':>8} {'DINO-C':>8}")
print("-" * 65)
for r in results:
    def fmt(v):
        return f"{v:.4f}" if v is not None else "  —   "
    print(f"{r['name']:<25} {fmt(r['clip_s']):>8} {fmt(r['lpips']):>8} {fmt(r['dino_s']):>8} {fmt(r['dino_c']):>8}")

with open(os.path.join(BASE, "_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {os.path.join(BASE, '_results.json')}")
