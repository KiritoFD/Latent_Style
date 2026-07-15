"""Extract destructive ablation results from all ablation directories."""
import json
from pathlib import Path

BASE = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp")
ABLATIONS = ["wo_flow", "wo_asg", "wo_wavelet", "wo_spectral_ode", "wo_endpoint_adain"]
DINO_DIR = BASE / "_dino_results"

print("=" * 80)
print("DESTRUCTIVE ABLATION RESULTS (D5-512)")
print("=" * 80)
print(f"{'Config':<25} {'CLIP-S':>8} {'LPIPS':>8} {'DINO-C':>8} {'DINO-S':>8} {'dCLIP-S':>8}")
print("-" * 80)

# Full model baseline
print(f"{'Full (WEAVE)':<25} {'0.7261':>8} {'0.3354':>8} {'0.7692':>8} {'0.4843':>8} {'--':>8}")
print("-" * 80)

for name in ABLATIONS:
    summary_path = BASE / "ablation_destructive" / name / "full_eval" / "epoch_0005" / "summary.json"
    dino_path = DINO_DIR / f"abl_{name}.json"

    clip_s = "N/A"
    lpips = "N/A"
    dino_c = "N/A"
    dino_s = "N/A"

    if summary_path.exists():
        s = json.load(open(summary_path))
        a = s.get("analysis", {})
        ap = a.get("all_pairs_overview", {})
        clip_s = f"{ap.get('clip_style', 0):.4f}" if ap.get('clip_style') is not None else "N/A"
        lpips = f"{ap.get('content_lpips', 0):.4f}" if ap.get('content_lpips') is not None else "N/A"

    if dino_path.exists():
        d = json.load(open(dino_path))
        dino_c = f"{d.get('dino_content', 0):.4f}" if d.get('dino_content') else "N/A"
        dino_s = f"{d.get('dino_style', 0):.4f}" if d.get('dino_style') else "N/A"

    # Compute delta
    delta = "--"
    if clip_s != "N/A":
        delta = f"{float(clip_s) - 0.7261:+.4f}"

    print(f"{name:<25} {clip_s:>8} {lpips:>8} {dino_c:>8} {dino_s:>8} {delta:>8}")

print("=" * 80)
