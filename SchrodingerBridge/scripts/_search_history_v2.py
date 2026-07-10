"""Search all summary.json for high clip_style (~0.74) and low lpips (~0.29).
Correctly extract from analysis.all_pairs_overview.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def extract_key_metrics(data):
    """Extract clip_style and content_lpips from analysis.all_pairs_overview."""
    result = {"clip_style": None, "lpips": None, "clip_content": None, "dino_sty": None, "dino_con": None}
    analysis = data.get("analysis", {})
    apo = analysis.get("all_pairs_overview", {})
    if isinstance(apo, dict):
        for k, v in apo.items():
            kl = k.lower()
            if "clip_style" in kl and isinstance(v, (int, float)) and result["clip_style"] is None:
                result["clip_style"] = float(v)
            elif "clip" in kl and "content" in kl and isinstance(v, (int, float)):
                result["clip_content"] = float(v)
            elif "lpips" in kl and "content" in kl and isinstance(v, (int, float)):
                result["lpips"] = float(v)
            elif "lpips" in kl and isinstance(v, (int, float)) and result["lpips"] is None:
                result["lpips"] = float(v)
            elif "dino" in kl and "sty" in kl and isinstance(v, (int, float)):
                result["dino_sty"] = float(v)
            elif "dino" in kl and "con" in kl and isinstance(v, (int, float)):
                result["dino_con"] = float(v)
    # Also check metrics_note as fallback
    mn = data.get("metrics_note", {})
    if isinstance(mn, dict):
        if result["clip_style"] is None and "clip_style" in mn:
            v = mn["clip_style"]
            if isinstance(v, (int, float)) and v < 1.0:
                result["clip_style"] = float(v)
    return result


def scan_dir(root):
    results = []
    for summary_path in Path(root).rglob("summary.json"):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            m = extract_key_metrics(data)
            if m["clip_style"] is not None or m["lpips"] is not None:
                results.append((str(summary_path), m))
        except Exception:
            pass
    return results


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "g:/GitHub/Latent_Style/SchrodingerBridge/exp"
    results = scan_dir(root)
    # Filter valid clip_style (not 1.0, not None)
    valid = [r for r in results if r[1]["clip_style"] is not None and r[1]["clip_style"] < 1.0]
    valid.sort(key=lambda x: -x[1]["clip_style"])
    print(f"=== TOTAL valid summaries (clip_style<1.0): {len(valid)} ===")
    print(f"{'clip_s':>8} {'lpips':>8} {'dino_sty':>8}  path")
    for path, m in valid[:40]:
        lp = f"{m['lpips']:.4f}" if m['lpips'] is not None else "  N/A "
        ds = f"{m['dino_sty']:.4f}" if m['dino_sty'] is not None else "  N/A "
        print(f"{m['clip_style']:>8.4f} {lp:>8} {ds:>8}  {path}")
    print()
    print("=== CANDIDATES: clip_s>=0.72 AND lpips<=0.32 ===")
    for path, m in valid:
        if m["clip_style"] >= 0.72 and m["lpips"] is not None and m["lpips"] <= 0.32:
            print(f"{m['clip_style']:.4f}  lpips={m['lpips']:.4f}  {path}")
