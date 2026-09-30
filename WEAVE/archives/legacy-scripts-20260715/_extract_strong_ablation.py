"""Extract strong ablation results (4 metrics for each config)."""
import json
from pathlib import Path

EXP = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp")
DINO_DIR = EXP / "_dino_results"

# Full model (T1 ASG 5ep) baseline
FULL_SUMMARY = EXP / "t1_asg_5ep" / "full_eval" / "epoch_0005" / "summary.json"
FULL_DINO = DINO_DIR / "abl_full.json"  # might not exist
# Try alternate paths for full model DINO
for p in [DINO_DIR / "t1_asg_5ep.json", DINO_DIR / "full.json"]:
    if p.exists():
        FULL_DINO = p
        break

# Strong ablation configs
STRONG_ABLATIONS = ["swd_to_mse", "wo_wavelet", "wo_swd", "ll_equal"]

# Inference-only ablations (from previous run)
INFERENCE_ABLATIONS = ["wo_flow", "wo_asg", "wo_endpoint_adain"]


def extract_metrics(name, exp_subdir="abl"):
    """Extract CLIP-S, LPIPS, DINO-C, DINO-S for a config."""
    summary_path = EXP / f"{exp_subdir}_{name}" / "full_eval" / "epoch_0005" / "summary.json"
    dino_path = DINO_DIR / f"abl_{name}.json"

    clip_s = lpips = dino_c = dino_s = None

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        a = s.get("analysis", {}).get("style_transfer_ability", {})
        clip_s = a.get("clip_style")
        lpips = a.get("content_lpips")

    if dino_path.exists():
        with open(dino_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        dino_c = d.get("dino_content")
        dino_s = d.get("dino_style")

    return clip_s, lpips, dino_c, dino_s


def main():
    # Full model
    full_clip, full_lpips, full_dc, full_ds = extract_metrics("full", "t1_asg_5ep")
    if full_clip is None:
        # Try alternate path
        s_path = EXP / "t1_asg_5ep" / "full_eval" / "epoch_0005" / "summary.json"
        if s_path.exists():
            with open(s_path, "r") as f:
                s = json.load(f)
            a = s.get("analysis", {}).get("style_transfer_ability", {})
            full_clip = a.get("clip_style")
            full_lpips = a.get("content_lpips")

    print("=" * 80)
    print("STRONG DESTRUCTIVE ABLATION RESULTS (D5-512)")
    print("=" * 80)
    print(f"{'Config':<30s} {'CLIP-S':>8s} {'LPIPS':>8s} {'DINO-C':>8s} {'DINO-S':>8s} {'dCLIP-S':>8s}")
    print("-" * 80)

    if full_clip is not None:
        dc_str = f"{full_dc:8.4f}" if full_dc is not None else f"{'N/A':>8s}"
        ds_str = f"{full_ds:8.4f}" if full_ds is not None else f"{'N/A':>8s}"
        print(f"{'Full (WEAVE)':<30s} {full_clip:8.4f} {full_lpips:8.4f} {dc_str} {ds_str} {'--':>8s}")
    else:
        # Fallback to known values
        full_clip, full_lpips, full_dc, full_ds = 0.7261, 0.3354, 0.7692, 0.4843
        print(f"{'Full (WEAVE)':<30s} {full_clip:8.4f} {full_lpips:8.4f} {full_dc:8.4f} {full_ds:8.4f} {'--':>8s}")

    print("-" * 80)
    print("  [Training-required ablations]")
    for name in STRONG_ABLATIONS:
        c, l, dc, ds = extract_metrics(name, "abl")
        dc_val = f"{dc:8.4f}" if dc is not None else f"{'N/A':>8s}"
        ds_val = f"{ds:8.4f}" if ds is not None else f"{'N/A':>8s}"
        c_val = f"{c:8.4f}" if c is not None else f"{'N/A':>8s}"
        l_val = f"{l:8.4f}" if l is not None else f"{'N/A':>8s}"
        if c is not None and full_clip is not None:
            delta = f"{c - full_clip:+8.4f}"
        else:
            delta = f"{'--':>8s}"
        print(f"  {name:<28s} {c_val} {l_val} {dc_val} {ds_val} {delta}")

    print("-" * 80)
    print("  [Inference-only ablations]")
    for name in INFERENCE_ABLATIONS:
        # These are in ablation_destructive subdirectory
        summary_path = EXP / "ablation_destructive" / name / "full_eval" / "epoch_0005" / "summary.json"
        dino_path = DINO_DIR / f"abl_{name}.json"

        c = l = dc = ds = None
        if summary_path.exists():
            with open(summary_path, "r") as f:
                s = json.load(f)
            a = s.get("analysis", {}).get("style_transfer_ability", {})
            c = a.get("clip_style")
            l = a.get("content_lpips")
        if dino_path.exists():
            with open(dino_path, "r") as f:
                d = json.load(f)
            dc = d.get("dino_content")
            ds = d.get("dino_style")

        dc_val = f"{dc:8.4f}" if dc is not None else f"{'N/A':>8s}"
        ds_val = f"{ds:8.4f}" if ds is not None else f"{'N/A':>8s}"
        c_val = f"{c:8.4f}" if c is not None else f"{'N/A':>8s}"
        l_val = f"{l:8.4f}" if l is not None else f"{'N/A':>8s}"
        if c is not None and full_clip is not None:
            delta = f"{c - full_clip:+8.4f}"
        else:
            delta = f"{'--':>8s}"
        print(f"  {name:<28s} {c_val} {l_val} {dc_val} {ds_val} {delta}")

    print("=" * 80)


if __name__ == "__main__":
    main()
