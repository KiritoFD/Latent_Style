"""Collect clip_style + LPIPS + IDT + FID metrics from all 31 X experiment summaries."""
import json
import re
from pathlib import Path

ROOT = Path(r"I:\Github\Latent_Style\SchrodingerBridge")
EXP_DIR = ROOT / "exp" / "628_ablation" / "destructive"
CONFIG_DIR = ROOT / "configs" / "ablations" / "628_destructive"

# Baseline reference (T5 ep7 from Phase 4)
BASELINE = {"clip_style": 0.7307, "lpips": 0.3403, "name": "T5_ep7_baseline"}


def parse_x_config(config_path: Path) -> tuple[str, float]:
    name = config_path.stem
    m = re.match(r"X(\d+)_(\w+?)_w(\d+)", name)
    if m:
        return m.group(2), float(m.group(3))
    if "combo" in name:
        if "all_w10" in name:
            return "all_combo", 10.0
        if "all_w50" in name:
            return "all_combo", 50.0
        if "content_w50" in name:
            return "content_combo", 50.0
        if "direction_w50" in name:
            return "direction_combo", 50.0
    return name, 0.0


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
    configs = sorted(CONFIG_DIR.glob("X*.json"), key=lambda p: p.stem)
    hdr = f"{'name':<28} {'loss':<18} {'w':>5} {'clip_s':>8} {'lpips':>8} {'ap_clip':>8} {'ap_lpip':>8} {'idt':>6} {'fid':>7}  dclip  dlpips"
    print(hdr)
    print("-" * len(hdr))
    print(f"{'[BASELINE T5 ep7]':<28} {'-':<18} {'-':>5} {BASELINE['clip_style']:>8.4f} {BASELINE['lpips']:>8.4f} {'-':>8} {'-':>8} {'-':>6} {'-':>7}  -      -")
    print("-" * len(hdr))

    results = []
    for cfg in configs:
        name = cfg.stem
        loss_name, weight = parse_x_config(cfg)
        m = read_summary(name)
        if m is None:
            print(f"{name:<28} {loss_name:<18} {weight:>5.0f}  MISSING")
            continue
        if "error" in m:
            print(f"{name:<28} {loss_name:<18} {weight:>5.0f}  ERROR: {m['error']}")
            continue
        d_clip = m["clip_style"] - BASELINE["clip_style"]
        d_lpips = m["content_lpips"] - BASELINE["lpips"]
        print(f"{name:<28} {loss_name:<18} {weight:>5.0f} {m['clip_style']:>8.4f} {m['content_lpips']:>8.4f} "
              f"{m['allpairs_clip']:>8.4f} {m['allpairs_lpips']:>8.4f} {m['idt_clip']:>6.3f} {m['fid']:>7.2f}  {d_clip:+.4f} {d_lpips:+.4f}")
        results.append({"name": name, "loss": loss_name, "weight": weight, **m})

    out_path = ROOT / "exp" / "628_ablation" / "destructive_logs" / "eval_results_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"baseline": BASELINE, "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Pareto analysis
    print("\n=== PARETO FRONT (clip_style desc, lpips asc) ===")
    valid = [r for r in results if r.get("clip_style", 0) > 0]
    # Pareto: a point is dominated if another has both higher clip AND lower lpips
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
    print(f"{'name':<28} {'loss':<18} {'w':>5} {'clip_s':>8} {'lpips':>8}  vs_base_clip  vs_base_lpips")
    print(f"{'[BASELINE]':<28} {'-':<18} {'-':>5} {BASELINE['clip_style']:>8.4f} {BASELINE['lpips']:>8.4f}  0  0")
    for r in pareto:
        print(f"{r['name']:<28} {r['loss']:<18} {r['weight']:>5.0f} {r['clip_style']:>8.4f} {r['content_lpips']:>8.4f}  "
              f"{r['clip_style']-BASELINE['clip_style']:+.4f}  {r['content_lpips']-BASELINE['lpips']:+.4f}")

    # Best per loss type
    print("\n=== BEST clip_style PER LOSS (single-loss only, no combos) ===")
    by_loss = {}
    for r in results:
        if "combo" in r["loss"]:
            continue
        by_loss.setdefault(r["loss"], []).append(r)
    print(f"{'loss':<14} {'best_clip':>10} {'@w':>5}  {'best_lpips':>11} {'@w':>5}  {'n':>3}")
    for loss, rs in sorted(by_loss.items()):
        bc = max(rs, key=lambda x: x["clip_style"])
        bl = min(rs, key=lambda x: x["content_lpips"])
        print(f"{loss:<14} {bc['clip_style']:>10.4f} {bc['weight']:>5.0f}  {bl['content_lpips']:>11.4f} {bl['weight']:>5.0f}  {len(rs):>3}")


if __name__ == "__main__":
    main()
