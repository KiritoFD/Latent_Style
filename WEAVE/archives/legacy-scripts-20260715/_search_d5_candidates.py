"""Search all local summary.json for clip_style>=0.70 on D5 (5-style) test dirs.

D5 = distinct5_512, wikiart_distinct5_samam_512_classview, wikiart512_5style.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

D5_MARKERS = ["distinct5", "wikiart512_5style", "classview", "5style", "d5"]


def is_d5_test(test_dir):
    if not test_dir:
        return False
    td = str(test_dir).lower().replace("\\", "/")
    return any(m in td for m in D5_MARKERS)


def extract(data):
    apo = data.get("analysis", {}).get("all_pairs_overview", {})
    cs = apo.get("clip_style")
    lp = apo.get("content_lpips")
    if cs is None or cs >= 1.0:
        return None, None
    return cs, lp


def scan(root):
    results = []
    for sp in Path(root).rglob("summary.json"):
        try:
            with open(sp, "r", encoding="utf-8") as f:
                data = json.load(f)
            cs, lp = extract(data)
            if cs is None:
                continue
            # detect D5
            settings = data.get("settings", {})
            test_dir = settings.get("test_image_dir", settings.get("test_dir", ""))
            intro = settings.get("introstyle_style_bank_root", "")
            is_d5 = is_d5_test(test_dir) or is_d5_test(intro)
            # matrix_breakdown style count
            mb = data.get("matrix_breakdown", {})
            n_styles = len(mb) if isinstance(mb, dict) else 0
            results.append({
                "path": str(sp),
                "clip_s": float(cs),
                "lpips": float(lp) if lp is not None else None,
                "test_dir": test_dir,
                "intro_root": intro,
                "is_d5": is_d5,
                "n_styles": n_styles,
            })
        except Exception:
            pass
    return results


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "g:/GitHub/Latent_Style/SchrodingerBridge/exp"
    results = scan(root)
    # Sort by clip_s descending
    results.sort(key=lambda x: -x["clip_s"])
    print(f"=== TOTAL: {len(results)} summaries with valid clip_s ===\n")
    print(f"{'clip_s':>8} {'lpips':>8} {'d5?':>4} {'ns':>3}  test_dir/intro_root  path")
    for r in results[:60]:
        lp = f"{r['lpips']:.4f}" if r['lpips'] is not None else "  N/A "
        d5 = "Y" if r["is_d5"] else "n"
        td = str(r["test_dir"] or r["intro_root"] or "")
        td_short = td.replace("g:/GitHub/Latent_Style/Dataset/", ".../").replace("G:\\GitHub\\Latent_Style\\Dataset\\", "...\\")
        print(f"{r['clip_s']:>8.4f} {lp:>8} {d5:>4} {r['n_styles']:>3}  {td_short}")
        print(f"{'':>34}  {r['path']}")
    print()
    print("=== D5 CANDIDATES (is_d5=True, clip_s>=0.70) ===")
    for r in results:
        if r["is_d5"] and r["clip_s"] >= 0.70:
            lp = f"{r['lpips']:.4f}" if r['lpips'] is not None else "N/A"
            print(f"  clip={r['clip_s']:.4f} lpips={lp}  ns={r['n_styles']}  {r['path']}")
            print(f"    test_dir={r['test_dir']}")
    print()
    print("=== D5 CANDIDATES (n_styles==5, clip_s>=0.70) ===")
    for r in results:
        if r["n_styles"] == 5 and r["clip_s"] >= 0.70:
            lp = f"{r['lpips']:.4f}" if r['lpips'] is not None else "N/A"
            d5 = "Y" if r["is_d5"] else "n"
            print(f"  clip={r['clip_s']:.4f} lpips={lp} d5={d5}  {r['path']}")
            print(f"    test_dir={r['test_dir']}  intro={r['intro_root']}")
