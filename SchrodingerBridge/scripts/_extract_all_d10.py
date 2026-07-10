"""Extract D10 metrics from remote experiment results.
Reads summary.json (CLIP-S, 1-LPIPS) and dino_results JSON (DINO-sty/con/str).
Usage: python _extract_all_d10.py
"""
import json
import os
from pathlib import Path

REMOTE_BASE = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
EXPS = ["d10a_dim48_15ep", "d10b_dim48_gate05_15ep", "d10c_dim32_15ep"]
BASELINE = {"clip_s": 0.7167, "1_LPIPS": 0.7010, "dino_sty": 0.4762, "dino_con": 0.8052, "dino_str": 0.0243}

def extract_exp(exp_name):
    """Extract metrics for one experiment."""
    exp_dir = Path(REMOTE_BASE) / exp_name
    summary_path = exp_dir / "full_eval" / "epoch_0015" / "summary.json"
    dino_path = Path(REMOTE_BASE) / "_dino_results" / f"{exp_name}.json"

    result = {"exp": exp_name, "found": False}

    # Check if files exist (local check; will be run on remote)
    if not summary_path.exists():
        result["error"] = f"summary.json not found: {summary_path}"
        return result
    if not dino_path.exists():
        result["error"] = f"dino_results not found: {dino_path}"
        return result

    # Extract CLIP-S and 1-LPIPS from summary.json
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    overview = summary.get("analysis", {}).get("all_pairs_overview", {})
    clip_style = overview.get("clip_style", 0.0)
    lpips_raw = overview.get("content_lpips", 1.0)
    one_minus_lpips = 1.0 - lpips_raw

    # Extract DINO metrics from dino_results JSON
    with open(dino_path, "r", encoding="utf-8") as f:
        dino = json.load(f)
    dino_style = dino.get("dino_style", 0.0)
    dino_content = dino.get("dino_content", 0.0)
    dino_structure = dino.get("dino_structure", 0.0)

    result.update({
        "found": True,
        "clip_s": round(clip_style, 4),
        "1_LPIPS": round(one_minus_lpips, 4),
        "dino_sty": round(dino_style, 4),
        "dino_con": round(dino_content, 4),
        "dino_str": round(dino_structure, 4),
        "delta_clip_s": round(clip_style - BASELINE["clip_s"], 4),
        "delta_dino_sty": round(dino_style - BASELINE["dino_sty"], 4),
        "delta_1_LPIPS": round(one_minus_lpips - BASELINE["1_LPIPS"], 4),
        "delta_dino_con": round(dino_content - BASELINE["dino_con"], 4),
    })
    return result


def main():
    print("=" * 90)
    print(f"{'config':<25} {'clip_s':>8} {'1-LPIPS':>9} {'dino_sty':>10} {'dino_con':>10} {'dino_str':>10} {'Δsty':>8}")
    print("-" * 90)
    print(f"{'hp baseline':<25} {0.7167:>8.4f} {0.7010:>9.4f} {0.4762:>10.4f} {0.8052:>10.4f} {0.0243:>10.4f} {0.0:>8.4f}")
    print("-" * 90)

    all_results = []
    for exp in EXPS:
        r = extract_exp(exp)
        if r.get("found"):
            print(f"{exp:<25} {r['clip_s']:>8.4f} {r['1_LPIPS']:>9.4f} {r['dino_sty']:>10.4f} {r['dino_con']:>10.4f} {r['dino_str']:>10.4f} {r['delta_dino_sty']:>+8.4f}")
            all_results.append(r)
        else:
            print(f"{exp:<25} ERROR: {r.get('error', 'unknown')}")

    print("=" * 90)
    print(f"\nExtracted {len(all_results)}/{len(EXPS)} experiments.")

    # Save results as JSON for state file update
    if all_results:
        out_path = Path("_d10_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
