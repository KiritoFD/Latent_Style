import json
import os

base = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
dino_base = r"I:\Github\Latent_Style\SchrodingerBridge\state\dino"

experiments = [
    ("refactor_minimal_baseline", "refactor_minimal_baseline"),
    ("refactor_clean_baseline", "refactor_clean_baseline"),
    ("ablation_destructive/wo_asg", "wo_asg"),
    ("ablation_destructive/wo_endpoint_adain", "wo_endpoint_adain"),
    ("ablation_destructive/wo_flow", "wo_flow"),
    ("ablation_destructive/wo_spectral_ode", "wo_spectral_ode"),
    ("ablation_destructive/wo_wavelet", "wo_wavelet"),
]

print(f"{'Experiment':<30} {'CLIP-S':>8} {'LPIPS':>8} {'DINO-S':>8} {'DINO-C':>8}")
print("-" * 70)

for exp_dir, name in experiments:
    clip_s = lpips = dino_s = dino_c = "N/A"
    # CLIP-S + LPIPS
    summary_path = os.path.join(base, exp_dir, "full_eval", "epoch_0005", "summary.json")
    if not os.path.exists(summary_path):
        summary_path = os.path.join(base, exp_dir, "eval", "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data = json.load(f)
        ov = data.get("analysis", {}).get("all_pairs_overview", {})
        clip_s = f"{ov.get('clip_style', 0):.4f}"
        lpips = f"{ov.get('content_lpips', 0):.4f}"
    # DINO
    dino_path = os.path.join(dino_base, f"D5-512__{name}.json")
    if os.path.exists(dino_path):
        with open(dino_path) as f:
            d = json.load(f)
        dino_s = f"{d.get('dino_style', 0):.4f}"
        dino_c = f"{d.get('dino_content', 0):.4f}"
    print(f"{name:<30} {clip_s:>8} {lpips:>8} {dino_s:>8} {dino_c:>8}")

print("\n=== DINO files available ===")
for f in sorted(os.listdir(dino_base)):
    if "D5-512" in f:
        print(f"  {f}")
