"""Compute CFSD (Content Fidelity Style Distance) from existing CLIP-S and LPIPS values.

Definition (project-internal, no standard public reference found):
  CF_distance = LPIPS          (content distance, lower = better content fidelity)
  SD           = 1 - CLIP-S     (style distance, lower = better style match)
  CFSD         = (1 + SD) * (1 + CF_distance) = (2 - CLIP-S) * (1 + LPIPS)
                 lower = better (symmetric to ArtFID = (1+FID)*(1+LPIPS))

This metric rewards methods that simultaneously preserve content (low LPIPS)
and match target style (high CLIP-S). It complements ArtFID (which uses
distributional FID for style) by using semantic CLIP-S for style.
"""
import json
from pathlib import Path

# Extracted from paper.tex main table (lines 319-328)
# Format: (method, D5_CLIP_S, D5_LPIPS, P256_CLIP_S, P256_LPIPS, R5_CLIP_S, R5_LPIPS)
methods = [
    ("identity",  0.6933, 0.0000, 0.6630, 0.0000, 0.7312, 0.0000),
    ("adain",     0.6679, 0.7425, 0.6656, 0.7357, 0.7276, 0.6795),
    ("wct",       0.7063, 0.6348, 0.6878, 0.7379, 0.7308, 0.6876),
    ("sdturbo",   0.6933, 0.0033, 0.6744, 0.6031, 0.7671, 0.4488),
    ("cut",       0.7137, 0.3743, None,   None,   0.7096, 0.6198),
    ("samst",     0.6183, 0.7490, 0.7092, 0.3981, 0.6669, 0.6118),
    ("samam",     0.5816, 0.2434, 0.6768, 0.2052, 0.7124, 0.2268),
    ("seedream",  0.7198, 0.4767, 0.7515, 0.2270, None,   None),
    ("weave",     0.7213, 0.2868, 0.6826, 0.2031, 0.7434, 0.2904),
]


def compute_cfsd(clip_s: float, lpips: float) -> float:
    """CFSD = (2 - CLIP-S) * (1 + LPIPS). Lower is better."""
    sd = 1.0 - clip_s
    cf_dist = lpips
    return (1.0 + sd) * (1.0 + cf_dist)


def main():
    datasets = ["D5-512", "P256", "R5-WikiArt"]
    results = {ds: {} for ds in datasets}

    print("=" * 80)
    print("CFSD (Content Fidelity Style Distance) - Lower is better")
    print("Formula: (2 - CLIP-S) * (1 + LPIPS)")
    print("=" * 80)

    for ds_idx, ds in enumerate(datasets):
        print(f"\n[{ds}]")
        print(f"  {'Method':<12} {'CLIP-S':>8} {'LPIPS':>8} {'CFSD':>8}")
        print("  " + "-" * 40)

        for method in methods:
            name = method[0]
            clip_s = method[1 + ds_idx * 2]
            lpips_val = method[2 + ds_idx * 2]

            if clip_s is None or lpips_val is None:
                print(f"  {name:<12} {'--':>8} {'--':>8} {'--':>8}")
                results[ds][name] = None
                continue

            cfsd = compute_cfsd(clip_s, lpips_val)
            print(f"  {name:<12} {clip_s:>8.4f} {lpips_val:>8.4f} {cfsd:>8.4f}")
            results[ds][name] = {
                "clip_s": clip_s,
                "lpips": lpips_val,
                "cfsd": cfsd,
            }

    # Summary ranking
    print("\n" + "=" * 80)
    print("CFSD Ranking (lower = better)")
    print("=" * 80)
    for ds in datasets:
        print(f"\n[{ds}]")
        valid = [(name, results[ds][name]["cfsd"]) for name in [m[0] for m in methods] if results[ds].get(name)]
        valid.sort(key=lambda x: x[1])
        for rank, (name, cfsd) in enumerate(valid, 1):
            marker = " ***" if name == "weave" else ""
            print(f"  {rank}. {name:<12} {cfsd:.4f}{marker}")

    out_path = Path("g:/GitHub/Latent_Style/SchrodingerBridge/results/_cfsd_summary.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
