"""Extract aggregate metrics (ALL-PAIRS / TRANSFER / IDENTITY) from a metrics.csv file.

Usage:
    python _extract_any_metrics.py <metrics.csv>
    python _extract_any_metrics.py I:\\...\\baseline_wikiarts15\\identity\\metrics.csv
"""
import csv
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python _extract_any_metrics.py <metrics.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
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

    if not rows:
        print(f"No valid rows in {csv_path}")
        sys.exit(1)

    print(f"=== {csv_path.name} ===")
    print(f"total rows: {len(rows)}")

    # All-pairs
    all_clip = sum(r["clip_s"] for r in rows) / len(rows)
    all_lpips = sum(r["lpips"] for r in rows) / len(rows)
    print(f"ALL-PAIRS:    count={len(rows)}, CLIP-S={all_clip:.4f}, LPIPS={all_lpips:.4f}")

    # Transfer (src != tgt)
    transfer = [r for r in rows if r["src_style"] != r["tgt_style"]]
    if transfer:
        t_clip = sum(r["clip_s"] for r in transfer) / len(transfer)
        t_lpips = sum(r["lpips"] for r in transfer) / len(transfer)
        print(f"TRANSFER:     count={len(transfer)}, CLIP-S={t_clip:.4f}, LPIPS={t_lpips:.4f}")
    else:
        t_clip = t_lpips = 0.0

    # Identity
    ident = [r for r in rows if r["src_style"] == r["tgt_style"]]
    if ident:
        i_clip = sum(r["clip_s"] for r in ident) / len(ident)
        i_lpips = sum(r["lpips"] for r in ident) / len(ident)
        print(f"IDENTITY:     count={len(ident)}, CLIP-S={i_clip:.4f}, LPIPS={i_lpips:.4f}")
    else:
        i_clip = i_lpips = 0.0

    out = {
        "csv": str(csv_path),
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
    print("\n=== JSON ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
