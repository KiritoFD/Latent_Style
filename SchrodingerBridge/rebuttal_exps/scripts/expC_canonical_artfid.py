"""Exp C (P0-A): Canonical ArtFID audit using existing result packets.

Per remaining_experiments_plan.md P0-A:
- Use only methods with 750/750 manifest consistency for the direct chart.
- IDT, WEAVE, SaMam are canonical (jaccard=1.0 with canonical manifest).
- Seedream 4.5 is 95% matching (720/750); include as supplementary reference.
- Z-STAR and StyleAligned only match 15/750 (jaccard=0.0101); EXCLUDE from
  direct chart, document as historical non-comparable.

This script reads the existing aaai2027_v4/fig_data/artfid_d5_audit.json
(which was already computed from results/_artfid_details.json) and the
artfid_component_table.csv (per-style breakdown), and produces a canonical
summary CSV + JSON. It does NOT regenerate any images.
"""
import csv
import json
from pathlib import Path

WEAVE_ROOT = Path(r"I:\Github\Latent_Style\WEAVE")  # remote (for reference)
LOCAL_ROOT = Path(r"g:\GitHub\Latent_Style\WEAVE")
OUTPUT_DIR = Path(r"g:\GitHub\Latent_Style\SchrodingerBridge\rebuttal_exps\experiments\rebuttal_20260716\expC_canonical_artfid")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Source data (already computed)
AUDIT_JSON = LOCAL_ROOT / "aaai2027_v4" / "fig_data" / "artfid_d5_audit.json"
COMPONENT_CSV = Path(r"g:\GitHub\Latent_Style\SchrodingerBridge\rebuttal_exps\experiments\rebuttal_20260716\artfid\artfid_component_table.csv")

# Canonical status from artfid_audit_summary.json
CANONICAL_METHODS = {"IDT", "WEAVE", "SaMam"}  # 750/750 match
REFERENCE_METHODS = {"Seedream 4.5"}           # 720/750 match (95%)
EXCLUDED_METHODS = {"Z-STAR", "StyleAligned"}   # 15/750 match (Random20 manifest)

TARGET_STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def main():
    print("=" * 70)
    print("Exp C (P0-A): Canonical ArtFID Audit")
    print("=" * 70)

    # Load audit JSON (aggregate values)
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    methods = audit["methods"]
    print(f"\nLoaded audit: {AUDIT_JSON}")
    print(f"Protocol: {audit.get('protocol', 'unknown')}")
    print(f"Methods: {list(methods.keys())}")

    # Load component CSV (per-style breakdown)
    per_style_rows = []
    if COMPONENT_CSV.exists():
        with open(COMPONENT_CSV, "r", encoding="utf-8-sig") as f:
            per_style_rows = list(csv.DictReader(f))
        print(f"\nLoaded per-style CSV: {COMPONENT_CSV} ({len(per_style_rows)} rows)")
    else:
        print(f"WARN: {COMPONENT_CSV} not found, per-style breakdown unavailable")

    # Build rows
    rows = []
    summary = {
        "protocol": audit.get("protocol"),
        "dataset": audit.get("dataset"),
        "canonical_methods": sorted(CANONICAL_METHODS),
        "reference_methods": sorted(REFERENCE_METHODS),
        "excluded_methods": sorted(EXCLUDED_METHODS),
        "per_method": {},
        "per_style": {},
    }

    # Aggregate rows
    for method, data in methods.items():
        canonical = method in CANONICAL_METHODS
        reference = method in REFERENCE_METHODS
        excluded = method in EXCLUDED_METHODS
        row = {
            "method": method,
            "target_style": "ALL",
            "count": data.get("count", 0),
            "raw_fid": data.get("fid"),
            "source_lpips": data.get("lpips"),
            "artfid": data.get("artfid"),
            "canonical": canonical,
            "reference": reference,
            "excluded": excluded,
            "extra": {k: v for k, v in data.items()
                      if k not in {"artfid", "fid", "lpips", "count"}},
        }
        rows.append(row)
        summary["per_method"][method] = row
        tag = "canonical" if canonical else ("reference" if reference else "excluded")
        print(f"  {method:<18} FID={data.get('fid'):.2f}, LPIPS={data.get('lpips'):.4f}, "
              f"ArtFID={data.get('artfid'):.2f}, count={data.get('count', 0)} [{tag}]")

    # Per-style rows (from component CSV)
    for r in per_style_rows:
        method = r["method"]
        style = r["target_style"]
        if style == "ALL":
            continue
        try:
            count = int(r["count"])
            raw_fid = float(r["raw_fid"])
            source_lpips = float(r["source_lpips"])
            artfid = float(r["artfid"])
        except (ValueError, KeyError):
            continue
        row = {
            "method": method,
            "target_style": style,
            "count": count,
            "raw_fid": raw_fid,
            "source_lpips": source_lpips,
            "artfid": artfid,
            "canonical": method in CANONICAL_METHODS,
            "reference": method in REFERENCE_METHODS,
            "excluded": method in EXCLUDED_METHODS,
        }
        rows.append(row)
        summary["per_style"].setdefault(method, {})[style] = row

    # Write CSV
    csv_path = OUTPUT_DIR / "canonical_artfid.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "method", "target_style", "count", "raw_fid", "source_lpips",
            "artfid", "canonical", "reference", "excluded"
        ])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"\nCSV saved: {csv_path}")

    # Write JSON
    json_path = OUTPUT_DIR / "canonical_artfid.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"JSON saved: {json_path}")

    # Print canonical chart (direct comparison)
    print("\n" + "=" * 70)
    print("CANONICAL ArtFID CHART (direct comparison, main paper)")
    print("=" * 70)
    print(f"  {'Method':<18} {'Count':<8} {'Raw FID':<12} {'Src LPIPS':<12} {'ArtFID':<12}")
    for r in rows:
        if r["target_style"] == "ALL" and (r["canonical"] or r["reference"]):
            tag = "canonical" if r["canonical"] else "reference"
            print(f"  {r['method']:<18} {r['count']:<8} {r['raw_fid']:<12.2f} "
                  f"{r['source_lpips']:<12.4f} {r['artfid']:<12.2f}  [{tag}]")

    print("\nEXCLUDED from direct chart (different source manifest, historical only):")
    for r in rows:
        if r["target_style"] == "ALL" and r["excluded"]:
            print(f"  {r['method']:<18} {r['count']:<8} {r['raw_fid']:<12.2f} "
                  f"{r['source_lpips']:<12.4f} {r['artfid']:<12.2f}  [excluded]")

    # Print per-style canonical chart
    print("\n" + "=" * 70)
    print("PER-STYLE ArtFID (canonical methods only)")
    print("=" * 70)
    header = f"  {'Style':<20}"
    for m in sorted(CANONICAL_METHODS):
        header += f" {m[:10]:<12}"
    print(header)
    for style in TARGET_STYLES:
        line = f"  {style:<20}"
        for m in sorted(CANONICAL_METHODS):
            v = summary["per_style"].get(m, {}).get(style, {}).get("artfid")
            line += f" {v:<12.2f}" if v is not None else f" {'N/A':<12}"
        print(line)

    print("\nEXPC_EXIT=0")


if __name__ == "__main__":
    main()
