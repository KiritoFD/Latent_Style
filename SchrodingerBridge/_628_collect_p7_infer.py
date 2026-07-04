"""Collect Phase 7C inference ablation results from infer_ablation dir."""
import json
from pathlib import Path

ROOT = Path(r"I:\Github\Latent_Style\SchrodingerBridge")
INFER_DIR = ROOT / "exp" / "628_ablation" / "infer_ablation"
STEPS_DIR = ROOT / "exp" / "628_ablation" / "infer_ablation_p7"

BASELINE = {"clip": 0.7307, "lpips": 0.3403, "name": "T5_ep7_baseline"}


def collect_infer_ablation():
    results = []
    for json_path in sorted(INFER_DIR.glob("P7*.json")):
        if json_path.name.endswith("_override.json"):
            continue
        try:
            with json_path.open("r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception as e:
            print(f"ERROR reading {json_path.name}: {e}")
            continue
        name = rec.get("exp_name", json_path.stem)
        m = rec.get("metrics", {})
        c = m.get("allpairs_clip_style") or m.get("transfer_clip_style")
        l = m.get("allpairs_content_lpips") or m.get("transfer_content_lpips")
        overrides = rec.get("overrides", {})
        results.append({
            "name": name,
            "clip": c,
            "lpips": l,
            "overrides": overrides,
        })
    return results


def collect_steps_strength():
    results = []
    if not STEPS_DIR.is_dir():
        return results
    for exp_dir in sorted(STEPS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        summary = exp_dir / "summary.json"
        if not summary.is_file():
            continue
        try:
            with summary.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        a = data.get("analysis", {})
        transfer = a.get("style_transfer_ability", {})
        allpairs = a.get("all_pairs_overview", {})
        c = transfer.get("clip_style") or allpairs.get("clip_style")
        l = transfer.get("content_lpips") or allpairs.get("content_lpips")
        results.append({"name": exp_dir.name, "clip": c, "lpips": l, "overrides": {}})
    return results


def main():
    print("=" * 80)
    print("Phase 7C Inference Ablation Results")
    print(f"Baseline: clip={BASELINE['clip']} lpips={BASELINE['lpips']}")
    print("=" * 80)

    results = collect_infer_ablation()
    print(f"\n--- #1-#10 (from infer_ablation dir): {len(results)} experiments ---")
    print(f"{'Name':<28} {'clip':>8} {'lpips':>8} {'d_clip':>8} {'d_lpips':>8}  overrides")
    for r in results:
        c = r["clip"]
        l = r["lpips"]
        dc = (c - BASELINE["clip"]) if c is not None else None
        dl = (l - BASELINE["lpips"]) if l is not None else None
        dc_s = f"{dc:+.4f}" if dc is not None else "N/A"
        dl_s = f"{dl:+.4f}" if dl is not None else "N/A"
        c_s = f"{c:.4f}" if c is not None else "N/A"
        l_s = f"{l:.4f}" if l is not None else "N/A"
        ov = r["overrides"]
        ov_s = ", ".join(f"{k}={v}" for k, v in ov.items())
        print(f"{r['name']:<28} {c_s:>8} {l_s:>8} {dc_s:>8} {dl_s:>8}  {ov_s}")

    ss_results = collect_steps_strength()
    print(f"\n--- #11-#12 (from infer_ablation_p7 dir): {len(ss_results)} experiments ---")
    for r in ss_results:
        c = r["clip"]
        l = r["lpips"]
        dc = (c - BASELINE["clip"]) if c is not None else None
        dl = (l - BASELINE["lpips"]) if l is not None else None
        dc_s = f"{dc:+.4f}" if dc is not None else "N/A"
        dl_s = f"{dl:+.4f}" if dl is not None else "N/A"
        c_s = f"{c:.4f}" if c is not None else "N/A"
        l_s = f"{l:.4f}" if l is not None else "N/A"
        print(f"{r['name']:<28} {c_s:>8} {l_s:>8} {dc_s:>8} {dl_s:>8}")


if __name__ == "__main__":
    main()
