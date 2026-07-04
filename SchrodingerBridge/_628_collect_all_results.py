"""Collect clip_style + LPIPS + IDT + FID metrics from ALL 159 experiment summaries.
Group results by series (D/L/E/P/X) and output comprehensive report.
"""
import json
import re
from pathlib import Path

ROOT = Path(r"I:\Github\Latent_Style\SchrodingerBridge")
EXP_DIR = ROOT / "exp" / "628_ablation" / "destructive"
CONFIG_DIR = ROOT / "configs" / "ablations" / "628_destructive"

# Baseline reference (T5 ep7 from Phase 4)
BASELINE = {"clip_style": 0.7307, "lpips": 0.3403, "name": "T5_ep7_baseline"}


def parse_config_name(config_path: Path) -> tuple[str, str, float]:
    """Extract series prefix, loss/component name, weight from config filename."""
    name = config_path.stem
    # X10_contrast_w10
    m = re.match(r"X(\d+)_(\w+?)_w(\d+)", name)
    if m:
        return "X", m.group(2), float(m.group(3))
    # D1_spectral_ode_off, D10_style_gate_film_only
    m = re.match(r"D(\d+)_(\w+)", name)
    if m:
        return "D", m.group(2), 0.0
    # E1_w_contrast_preserve
    m = re.match(r"E(\d+)_w_(\w+)", name)
    if m:
        return "E", m.group(2), 1.0  # E series default w=1.0
    # L1_no_endpoint_content
    m = re.match(r"L(\d+)_no_(\w+)", name)
    if m:
        return "L", m.group(2), 0.0  # L series disables (w=0)
    # P10_wkin_05, P11_sigma_000
    m = re.match(r"P(\d+)_(\w+?)_(\d+)", name)
    if m:
        return "P", m.group(2), float(m.group(3)) / 100.0  # encode weight as fraction
    return name[0], name, 0.0


def read_summary(exp_name: str) -> dict | None:
    summary_path = EXP_DIR / exp_name / "full_eval" / "epoch_0010" / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"error": str(e)}
    a = data.get("analysis", {})
    transfer = a.get("style_transfer_ability", {})
    allpairs = a.get("all_pairs_overview", {})
    idt = a.get("identity_reconstruction", {})
    return {
        "clip_style": float(transfer.get("clip_style", 0.0) or 0.0),
        "content_lpips": float(transfer.get("content_lpips", 0.0) or 0.0),
        "allpairs_clip": float(allpairs.get("clip_style", 0.0) or 0.0),
        "allpairs_lpips": float(allpairs.get("content_lpips", 0.0) or 0.0),
        "idt_clip": float(idt.get("clip_style", 0.0) or 0.0),
        "fid": float(allpairs.get("fid", 0.0) or 0.0),
        "delta_fid": float(allpairs.get("delta_fid", 0.0) or 0.0),
    }


def main():
    configs = sorted(CONFIG_DIR.glob("*.json"), key=lambda p: p.stem)
    print(f"# 628 Destructive Ablation - Full Results ({len(configs)} experiments)")
    print(f"# Baseline: T5 ep7 clip={BASELINE['clip_style']}, lpips={BASELINE['lpips']}")
    print()

    results = []
    for cfg in configs:
        name = cfg.stem
        series, loss_name, weight = parse_config_name(cfg)
        m = read_summary(name)
        if m is None:
            print(f"MISSING: {name}")
            continue
        if "error" in m:
            print(f"ERROR: {name}: {m['error']}")
            continue
        results.append({
            "name": name, "series": series, "loss": loss_name, "weight": weight, **m
        })

    # Save full JSON
    out_path = ROOT / "exp" / "628_ablation" / "destructive_logs" / "all_results_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"baseline": BASELINE, "results": results}, f, indent=2)
    print(f"Saved full JSON to {out_path}")
    print()

    # === Per-series report ===
    for series in ["D", "L", "E", "P", "X"]:
        sr = [r for r in results if r["series"] == series]
        if not sr:
            continue
        print(f"\n=== Series {series} ({len(sr)} experiments) ===")
        print(f"{'name':<32} {'loss':<22} {'clip_s':>8} {'lpips':>8} {'ap_clip':>8} {'ap_lpip':>8} {'idt':>6}  dclip  dlpips")
        print("-" * 130)
        for r in sorted(sr, key=lambda x: -x["clip_style"]):
            d_clip = r["clip_style"] - BASELINE["clip_style"]
            d_lpips = r["content_lpips"] - BASELINE["lpips"]
            print(f"{r['name']:<32} {r['loss']:<22} {r['clip_style']:>8.4f} {r['content_lpips']:>8.4f} "
                  f"{r['allpairs_clip']:>8.4f} {r['allpairs_lpips']:>8.4f} {r['idt_clip']:>6.3f}  {d_clip:+.4f} {d_lpips:+.4f}")

    # === Global Pareto Front ===
    print("\n=== GLOBAL PARETO FRONT (clip_style desc, lpips asc) ===")
    valid = [r for r in results if r.get("clip_style", 0) > 0]
    pareto = []
    for r in valid:
        dominated = False
        for o in valid:
            if o is r:
                continue
            if (o["clip_style"] > r["clip_style"] and o["content_lpips"] <= r["content_lpips"]) or \
               (o["clip_style"] >= r["clip_style"] and o["content_lpips"] < r["content_lpips"]):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    pareto.sort(key=lambda x: -x["clip_style"])
    print(f"{'name':<32} {'series':<6} {'clip_s':>8} {'lpips':>8}  dclip  dlpips")
    print(f"{'[BASELINE T5 ep7]':<32} {'-':<6} {BASELINE['clip_style']:>8.4f} {BASELINE['lpips']:>8.4f}  0  0")
    for r in pareto:
        print(f"{r['name']:<32} {r['series']:<6} {r['clip_style']:>8.4f} {r['content_lpips']:>8.4f}  "
              f"{r['clip_style']-BASELINE['clip_style']:+.4f}  {r['content_lpips']-BASELINE['lpips']:+.4f}")

    # === Top 10 clip_style (excluding baseline) ===
    print("\n=== TOP 10 clip_style (excluding baseline) ===")
    top_clip = sorted(results, key=lambda x: -x["clip_style"])[:10]
    print(f"{'name':<32} {'series':<6} {'clip_s':>8} {'lpips':>8}")
    for r in top_clip:
        print(f"{r['name']:<32} {r['series']:<6} {r['clip_style']:>8.4f} {r['content_lpips']:>8.4f}")

    # === Top 10 lowest LPIPS ===
    print("\n=== TOP 10 lowest LPIPS (excluding baseline) ===")
    top_lpips = sorted(results, key=lambda x: x["content_lpips"])[:10]
    print(f"{'name':<32} {'series':<6} {'clip_s':>8} {'lpips':>8}")
    for r in top_lpips:
        print(f"{r['name']:<32} {r['series']:<6} {r['clip_style']:>8.4f} {r['content_lpips']:>8.4f}")

    # === Best per series ===
    print("\n=== BEST clip_style PER SERIES ===")
    print(f"{'series':<6} {'best_clip':>10} {'name':<32}  {'best_lpips':>11} {'name':<32}")
    for series in ["D", "L", "E", "P", "X"]:
        sr = [r for r in results if r["series"] == series]
        if not sr:
            continue
        bc = max(sr, key=lambda x: x["clip_style"])
        bl = min(sr, key=lambda x: x["content_lpips"])
        print(f"{series:<6} {bc['clip_style']:>10.4f} {bc['name']:<32}  {bl['content_lpips']:>11.4f} {bl['name']:<32}")


if __name__ == "__main__":
    main()
