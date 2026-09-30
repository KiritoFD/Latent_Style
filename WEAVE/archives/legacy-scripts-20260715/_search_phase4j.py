"""Search phase4j1, phase4f, dwt_route related results on D5."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def extract(data):
    apo = data.get("analysis", {}).get("all_pairs_overview", {})
    cs = apo.get("clip_style")
    lp = apo.get("content_lpips")
    if cs is None or cs >= 1.0:
        return None, None
    return cs, lp

def scan(root, keywords):
    results = []
    for sp in Path(root).rglob("summary.json"):
        pstr = str(sp).lower()
        if not any(k in pstr for k in keywords):
            continue
        try:
            with open(sp, "r", encoding="utf-8") as f:
                data = json.load(f)
            cs, lp = extract(data)
            if cs is None:
                continue
            settings = data.get("settings", {})
            test_dir = settings.get("test_image_dir", "") or settings.get("test_dir", "") or settings.get("introstyle_style_bank_root", "")
            mb = data.get("matrix_breakdown", {})
            n_styles = len(mb) if isinstance(mb, dict) else 0
            results.append({
                "path": str(sp),
                "clip_s": float(cs),
                "lpips": float(lp) if lp is not None else None,
                "test_dir": str(test_dir),
                "n_styles": n_styles,
            })
        except Exception:
            pass
    return results

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "g:/GitHub/Latent_Style/SchrodingerBridge/exp"
    # Search for phase4j, phase4f, dwt_route, 4j1, 4f1, stochastic_dwt
    keywords = ["phase4j", "phase4f", "dwt_route", "stochastic_dwt", "4j1", "4f1", "4f.", "4j."]
    results = scan(root, keywords)
    results.sort(key=lambda x: -x["clip_s"])
    print(f"=== Phase4J/F related summaries: {len(results)} ===\n")
    for r in results[:30]:
        lp = f"{r['lpips']:.4f}" if r['lpips'] is not None else "N/A"
        td = r['test_dir'].replace("\\", "/")
        for prefix in ["g:/GitHub/Latent_Style/Dataset/", "I:/datasets/", "I:/Github/Latent_Style/", "F:/", "G:/GitHub/Latent_Style/Dataset/"]:
            td = td.replace(prefix, "")
        print(f"clip={r['clip_s']:.4f} lpips={lp} ns={r['n_styles']} test={td}")
        print(f"  {r['path']}")