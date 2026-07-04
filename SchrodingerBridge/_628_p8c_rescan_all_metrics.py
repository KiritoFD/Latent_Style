"""Phase 8C: Rescan ALL historical summary.json with CORRECT metrics.

Metric confusion discovery (Phase 8B):
- all_pairs_overview.clip_style (~0.73) includes identity pairs
- style_transfer_ability.clip_style (~0.70) excludes identity pairs (pure transfer)
- Historical baseline 0.7307 is all_pairs_overview
- Previous _read_summary prioritized style_transfer_ability (BUG)

This script scans all summary.json under exp/628_ablation/ and extracts BOTH
metrics to identify which experiments actually improved over baseline.

Usage (remote):
    python _628_p8c_rescan_all_metrics.py
"""
import json
import os
from pathlib import Path

ROOT = Path(r"I:\Github\Latent_Style\SchrodingerBridge")
ABLATION_ROOT = ROOT / "exp" / "628_ablation"
OUTPUT = ABLATION_ROOT / "p8c_rescan_results.json"

BASELINE_ALLPAIRS_CLIP = 0.7307
BASELINE_ALLPAIRS_LPIPS = 0.3403
BASELINE_TRANSFER_CLIP = 0.7016
BASELINE_TRANSFER_LPIPS = 0.3520


def extract_metrics(summary_path: Path) -> dict | None:
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    a = data.get("analysis", {}) or {}
    transfer = a.get("style_transfer_ability", {}) or {}
    allpairs = a.get("all_pairs_overview", {}) or {}
    return {
        "clip_allpairs": allpairs.get("clip_style"),
        "clip_transfer": transfer.get("clip_style"),
        "lpips_allpairs": allpairs.get("content_lpips"),
        "lpips_transfer": transfer.get("content_lpips"),
    }


def find_summaries(root: Path) -> list[tuple[str, Path]]:
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "summary.json" in filenames:
            rel = os.path.relpath(dirpath, root)
            results.append((rel, Path(dirpath) / "summary.json"))
    return results


def main():
    summaries = find_summaries(ABLATION_ROOT)
    print(f"[P8C] Found {len(summaries)} summary.json files", flush=True)

    all_results = []
    for rel, spath in summaries:
        m = extract_metrics(spath)
        if m is None:
            continue
        m["path"] = rel
        all_results.append(m)

    # Sort by all_pairs clip_style descending
    valid = [r for r in all_results if r.get("clip_allpairs") is not None]
    valid.sort(key=lambda r: r["clip_allpairs"], reverse=True)

    # Save full results
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump({
            "phase": "8C",
            "description": "Rescan all historical experiments with correct metrics",
            "baseline": {
                "allpairs_clip": BASELINE_ALLPAIRS_CLIP,
                "allpairs_lpips": BASELINE_ALLPAIRS_LPIPS,
                "transfer_clip": BASELINE_TRANSFER_CLIP,
                "transfer_lpips": BASELINE_TRANSFER_LPIPS,
            },
            "total_scanned": len(all_results),
            "valid_results": len(valid),
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"[P8C] Saved {len(all_results)} results to {OUTPUT}", flush=True)

    # Print top 20 by all_pairs clip_style
    print("\n=== TOP 20 by all_pairs_overview.clip_style ===", flush=True)
    print(f"{'clip_ap':>8} | {'lpips_ap':>8} | {'clip_tr':>8} | {'lpips_tr':>8} | path", flush=True)
    print("-" * 80, flush=True)
    print(f"{BASELINE_ALLPAIRS_CLIP:>8.4f} | {BASELINE_ALLPAIRS_LPIPS:>8.4f} | "
          f"{BASELINE_TRANSFER_CLIP:>8.4f} | {BASELINE_TRANSFER_LPIPS:>8.4f} | BASELINE (T5 ep7)", flush=True)
    for r in valid[:20]:
        ca = r["clip_allpairs"] or 0.0
        la = r["lpips_allpairs"] or 0.0
        ct = r["clip_transfer"] or 0.0
        lt = r["lpips_transfer"] or 0.0
        print(f"{ca:>8.4f} | {la:>8.4f} | {ct:>8.4f} | {lt:>8.4f} | {r['path']}", flush=True)

    # Print experiments that BEAT baseline on all_pairs clip_style
    beaters = [r for r in valid if r["clip_allpairs"] > BASELINE_ALLPAIRS_CLIP]
    print(f"\n=== Experiments BEATING baseline all_pairs clip ({BASELINE_ALLPAIRS_CLIP}) ===", flush=True)
    print(f"Count: {len(beaters)}", flush=True)
    for r in beaters:
        ca = r["clip_allpairs"] or 0.0
        la = r["lpips_allpairs"] or 0.0
        delta = ca - BASELINE_ALLPAIRS_CLIP
        print(f"  {ca:.4f} (+{delta:.4f}) | lpips={la:.4f} | {r['path']}", flush=True)

    # Print Pareto front (all_pairs metric)
    print("\n=== Pareto Front (all_pairs_overview metric) ===", flush=True)
    pareto = []
    for r in valid:
        ca = r["clip_allpairs"] or 0.0
        la = r["lpips_allpairs"] or 999.0
        dominated = False
        for r2 in valid:
            if r2 is r:
                continue
            ca2 = r2["clip_allpairs"] or 0.0
            la2 = r2["lpips_allpairs"] or 999.0
            if ca2 >= ca and la2 <= la and (ca2 > ca or la2 < la):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    pareto.sort(key=lambda r: r["lpips_allpairs"] or 999.0)
    for r in pareto:
        ca = r["clip_allpairs"] or 0.0
        la = r["lpips_allpairs"] or 0.0
        print(f"  clip={ca:.4f} | lpips={la:.4f} | {r['path']}", flush=True)


if __name__ == "__main__":
    main()
