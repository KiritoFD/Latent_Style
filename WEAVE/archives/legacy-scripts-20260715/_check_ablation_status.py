"""Check strong ablation experiment status and extract metrics."""
import json
import os

EXP_ROOT = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
ABLATIONS = ["swd_to_mse", "wo_wavelet", "wo_swd", "ll_equal"]

# Full model baseline (T1 ASG 5ep) - from prior evaluation
FULL = {"clip_s": 0.7261, "lpips": 0.3354, "dino_c": 0.7692, "dino_s": 0.4843}


def load_metrics(name):
    """Load CLIP-S/LPIPS from summary.json, DINO-C/DINO-S from _dino_results."""
    clip_s = lpips = dino_c = dino_s = None
    summ = os.path.join(EXP_ROOT, f"abl_{name}", "full_eval", "epoch_0005", "summary.json")
    if os.path.exists(summ):
        with open(summ, "r") as f:
            data = json.load(f)
        ov = data.get("analysis", {}).get("all_pairs_overview", {})
        clip_s = ov.get("clip_style")
        lpips = ov.get("content_lpips")

    dino = os.path.join(EXP_ROOT, "_dino_results", f"abl_{name}.json")
    if os.path.exists(dino):
        with open(dino, "r") as f:
            dd = json.load(f)
        # Check if this is a fresh DINO result (after 2026-07-11 15:25 for swd_to_mse)
        dino_c = dd.get("dino_content")
        dino_s = dd.get("dino_style")
    return clip_s, lpips, dino_c, dino_s


print("=" * 95)
print(f"{'Ablation':<18} {'CKPT':<6} {'EVAL':<6} {'DINO':<6}  {'CLIP-S':>8} {'LPIPS':>8} {'DINO-C':>8} {'DINO-S':>8}")
print("-" * 95)
print(f"{'Full (T1 ASG)':<18} {'OK':<6} {'OK':<6} {'OK':<6}  "
      f"{FULL['clip_s']:>8.4f} {FULL['lpips']:>8.4f} {FULL['dino_c']:>8.4f} {FULL['dino_s']:>8.4f}")

results = {}
for name in ABLATIONS:
    d = os.path.join(EXP_ROOT, f"abl_{name}")
    ckpt = os.path.join(d, "epoch_0005.pt")
    summ = os.path.join(d, "full_eval", "epoch_0005", "summary.json")
    dino = os.path.join(EXP_ROOT, "_dino_results", f"abl_{name}.json")

    ck_ok = "OK" if os.path.exists(ckpt) else "--"
    ev_ok = "OK" if os.path.exists(summ) else "--"
    dn_ok = "OK" if os.path.exists(dino) else "--"

    clip_s, lpips, dino_c, dino_s = load_metrics(name)
    results[name] = {"clip_s": clip_s, "lpips": lpips, "dino_c": dino_c, "dino_s": dino_s,
                     "ckpt": ck_ok, "eval": ev_ok, "dino": dn_ok}

    def fmt(v):
        return f"{v:>8.4f}" if v is not None else f"{'--':>8}"
    print(f"abl_{name:<14} {ck_ok:<6} {ev_ok:<6} {dn_ok:<6}  "
          f"{fmt(clip_s)} {fmt(lpips)} {fmt(dino_c)} {fmt(dino_s)}")

print("=" * 95)

# Check DINO file timestamps to verify they're fresh
print("\nDINO file timestamps (to verify freshness):")
for name in ABLATIONS:
    dino = os.path.join(EXP_ROOT, "_dino_results", f"abl_{name}.json")
    if os.path.exists(dino):
        mtime = os.path.getmtime(dino)
        import time
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        print(f"  abl_{name}: {ts}")
    else:
        print(f"  abl_{name}: not found")
