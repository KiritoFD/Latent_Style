"""Search all summary.json for high clip_s (~0.74) and low lpips (~0.29)."""
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def extract_metrics(d, found, path=""):
    """Recursively extract clip_style and content_lpips from nested dict."""
    if isinstance(d, dict):
        for k, v in d.items():
            kl = k.lower()
            if "clip" in kl and "style" in kl and isinstance(v, (int, float)):
                found.setdefault("clip_s", []).append((f"{path}.{k}", float(v)))
            if "lpips" in kl and "content" in kl and isinstance(v, (int, float)):
                found.setdefault("lpips", []).append((f"{path}.{k}", float(v)))
            if "lpips" in kl and isinstance(v, (int, float)) and "lpips" not in found:
                found.setdefault("lpips_any", []).append((f"{path}.{k}", float(v)))
            if "all_pairs" in kl and isinstance(v, dict):
                extract_metrics(v, found, f"{path}.{k}")
            elif isinstance(v, (dict, list)):
                extract_metrics(v, found, f"{path}.{k}")
    elif isinstance(d, list):
        for i, item in enumerate(d):
            extract_metrics(item, found, f"{path}[{i}]")


def scan_dir(root):
    results = []
    for summary_path in Path(root).rglob("summary.json"):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            found = {}
            extract_metrics(data, found)
            clip_s = None
            lpips = None
            if "clip_s" in found:
                clip_s = found["clip_s"][0][1]
            lpips_list = found.get("lpips", found.get("lpips_any", []))
            if lpips_list:
                lpips = lpips_list[0][1]
            if clip_s is not None or lpips is not None:
                results.append((str(summary_path), clip_s, lpips))
        except Exception:
            pass
    return results


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "g:/GitHub/Latent_Style/SchrodingerBridge/exp"
    results = scan_dir(root)
    # Sort by clip_s descending, show top candidates
    results_with_clip = [r for r in results if r[1] is not None]
    results_with_clip.sort(key=lambda x: -x[1])
    print(f"=== TOTAL summaries with metrics: {len(results_with_clip)} ===")
    print(f"=== TOP 30 by clip_s (looking for ~0.74 with lpips~0.29) ===")
    print(f"{'clip_s':>8} {'lpips':>8}  path")
    for path, clip_s, lpips in results_with_clip[:30]:
        lp_str = f"{lpips:.4f}" if lpips is not None else "  N/A "
        print(f"{clip_s:>8.4f} {lp_str:>8}  {path}")
    print()
    print("=== CANDIDATES: clip_s>=0.725 AND lpips<=0.31 ===")
    for path, clip_s, lpips in results_with_clip:
        if clip_s >= 0.725 and lpips is not None and lpips <= 0.31:
            print(f"{clip_s:.4f}  {lpips:.4f}  {path}")
