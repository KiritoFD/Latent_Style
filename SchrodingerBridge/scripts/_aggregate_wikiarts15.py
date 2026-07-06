"""Aggregate per-row metrics.csv for wikiarts-15 baselines AND WD-VF into CLIP-S / LPIPS means."""
import csv
import os

REPO = r"I:\Github\Latent_Style\SchrodingerBridge"

# Baselines
BASELINE_ROOT = os.path.join(REPO, "exp", "baseline_wikiarts15")
# WD-VF
WDVF_ROOT = os.path.join(REPO, "exp", "wikiarts15_eval")

def agg_csv(p, clip_col=4, lpips_col=8):
    if not os.path.exists(p):
        return None
    rows = list(csv.reader(open(p)))
    if not rows:
        return None
    n = len(rows)
    try:
        clip_vals = [float(r[clip_col]) for r in rows]
        lpips_vals = [float(r[lpips_col]) for r in rows]
        # also extract unique styles
        src_styles = set(r[0] for r in rows)
        tgt_styles = set(r[1] for r in rows)
        return {
            "n": n,
            "clip_s": sum(clip_vals) / n,
            "lpips": sum(lpips_vals) / n,
            "src_styles": sorted(src_styles),
            "tgt_styles": sorted(tgt_styles),
            "n_src": len(src_styles),
            "n_tgt": len(tgt_styles),
        }
    except (IndexError, ValueError) as e:
        return {"error": str(e), "first_row": rows[0][:12] if rows else None}

print("=== Baselines ===")
for m in ["identity", "adain", "wct"]:
    p = os.path.join(BASELINE_ROOT, m, "metrics.csv")
    r = agg_csv(p)
    if r is None:
        print(f"{m}: MISSING")
    elif "error" in r:
        print(f"{m}: parse error: {r['error']}, first_row: {r['first_row']}")
    else:
        print(f"{m}: n={r['n']} CLIP-S={r['clip_s']:.4f} LPIPS={r['lpips']:.4f} "
              f"src_styles={r['n_src']} tgt_styles={r['n_tgt']}")
        print(f"  src: {r['src_styles']}")

print()
print("=== WD-VF ===")
p = os.path.join(WDVF_ROOT, "metrics.csv")
r = agg_csv(p)
if r is None:
    print(f"WD-VF: MISSING")
elif "error" in r:
    print(f"WD-VF: parse error: {r['error']}, first_row: {r['first_row']}")
else:
    print(f"WD-VF: n={r['n']} CLIP-S={r['clip_s']:.4f} LPIPS={r['lpips']:.4f} "
          f"src_styles={r['n_src']} tgt_styles={r['n_tgt']}")
    print(f"  src: {r['src_styles']}")
    print(f"  tgt: {r['tgt_styles']}")
