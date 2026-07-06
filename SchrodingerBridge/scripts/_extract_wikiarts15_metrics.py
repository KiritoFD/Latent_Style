"""Extract WD-VF wikiarts-15 aggregate metrics from summary.json + metrics.csv on remote.
Run locally via ssh + python.
"""
import json
import csv
import sys
from pathlib import Path

SUM_PATH = r"I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts15_eval\summary.json"
CSV_PATH = r"I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts15_eval\metrics.csv"


def main():
    # Read summary.json
    with open(SUM_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    print("=== summary.json top-level keys ===")
    print(list(summary.keys()))

    print("\n=== analysis section ===")
    analysis = summary.get("analysis", {})
    print(json.dumps(analysis, indent=2, default=str)[:5000])

    print("\n=== idt_baselines section ===")
    idt = summary.get("idt_baselines", {})
    print(json.dumps(idt, indent=2, default=str)[:3000])

    # Compute from CSV directly
    print("\n=== Computed from CSV ===")
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "src_style": r["src_style"],
                    "tgt_style": r["tgt_style"],
                    "lpips": float(r["content_lpips"]),
                    "clip_s": float(r["clip_style"]),
                })
            except (ValueError, KeyError) as e:
                continue

    print(f"total rows: {len(rows)}")

    # All-pairs
    all_clip = sum(r["clip_s"] for r in rows) / len(rows)
    all_lpips = sum(r["lpips"] for r in rows) / len(rows)
    print(f"ALL-PAIRS:    count={len(rows)}, CLIP-S={all_clip:.4f}, LPIPS={all_lpips:.4f}")

    # Transfer (src != tgt)
    transfer = [r for r in rows if r["src_style"] != r["tgt_style"]]
    t_clip = sum(r["clip_s"] for r in transfer) / len(transfer)
    t_lpips = sum(r["lpips"] for r in transfer) / len(transfer)
    print(f"TRANSFER:     count={len(transfer)}, CLIP-S={t_clip:.4f}, LPIPS={t_lpips:.4f}")

    # Identity
    ident = [r for r in rows if r["src_style"] == r["tgt_style"]]
    i_clip = sum(r["clip_s"] for r in ident) / len(ident)
    i_lpips = sum(r["lpips"] for r in ident) / len(ident)
    print(f"IDENTITY:     count={len(ident)}, CLIP-S={i_clip:.4f}, LPIPS={i_lpips:.4f}")

    # Per-style transfer averages
    print("\n=== Per-style transfer CLIP-S (target style) ===")
    by_tgt = {}
    for r in transfer:
        by_tgt.setdefault(r["tgt_style"], []).append(r)
    for style in sorted(by_tgt.keys()):
        rs = by_tgt[style]
        c = sum(x["clip_s"] for x in rs) / len(rs)
        l = sum(x["lpips"] for x in rs) / len(rs)
        print(f"  {style:40s}  CLIP-S={c:.4f}  LPIPS={l:.4f}  n={len(rs)}")

    # Write final answer as JSON
    out = {
        "allpairs_clip_s": round(all_clip, 4),
        "allpairs_lpips": round(all_lpips, 4),
        "transfer_clip_s": round(t_clip, 4),
        "transfer_lpips": round(t_lpips, 4),
        "identity_clip_s": round(i_clip, 4),
        "identity_lpips": round(i_lpips, 4),
        "n_allpairs": len(rows),
        "n_transfer": len(transfer),
        "n_identity": len(ident),
    }
    print("\n=== FINAL METRICS JSON ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
