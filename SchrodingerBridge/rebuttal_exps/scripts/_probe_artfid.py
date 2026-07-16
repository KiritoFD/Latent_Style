"""Probe ArtFID data structure - write to file."""
import json
import sys
from pathlib import Path

OUT = Path("C:/Users/Administrator/_probe_artfid_out.txt")
lines = []

def log(msg=""):
    lines.append(str(msg))

paths = [
    "I:/Github/Latent_Style/WEAVE/docs/experiments/comparison_20260602/artfid_comparison_points.json",
    "I:/Github/Latent_Style/WEAVE/docs/experiments/comparison_20260602/artfid_comparison_points.csv",
    "I:/Github/Latent_Style/WEAVE/runs/submission/hf_oriented_internal_early_stop/artfid_oriented_e4_epoch_metrics.json",
    "I:/Github/Latent_Style/WEAVE/runs/submission/hf_oriented_internal_early_stop/artfid_oriented_e4_epoch_metrics.csv",
    "I:/Github/Latent_Style/WEAVE/exp/repro_weave_d5/summary.json",
    "I:/Github/Latent_Style/WEAVE/exp/repro_weave_d5/metrics.csv",
    "I:/Github/Latent_Style/WEAVE/exp/compat_baseline_after_cleanup_v2/summary.json",
    "I:/Github/Latent_Style/WEAVE/exp/target_style_baseline_p2a/summary.json",
    "I:/Github/Latent_Style/WEAVE/exp/target_style_baseline_r5/summary.json",
]

for p in paths:
    pp = Path(p)
    log(f"\n=== {p} ===")
    if not pp.exists():
        log("  MISSING")
        continue
    if pp.suffix == ".json":
        try:
            d = json.loads(pp.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(d, dict):
                log(f"  dict keys: {list(d.keys())[:30]}")
                artfid_keys = [k for k in d.keys() if "artfid" in k.lower() or "art_fid" in k.lower() or "fid" in k.lower()]
                log(f"  artfid-related keys: {artfid_keys}")
                for k in artfid_keys:
                    v = d[k]
                    if isinstance(v, (int, float, str)) or (isinstance(v, (list, dict)) and len(str(v)) < 500):
                        log(f"    {k} = {v}")
                    else:
                        log(f"    {k} = {type(v).__name__}, len={len(v) if hasattr(v, '__len__') else 'NA'}")
                for k, v in list(d.items())[:20]:
                    if isinstance(v, (int, float, str)):
                        s = str(v)
                        if len(s) < 80:
                            log(f"    {k} = {v}")
            else:
                log(f"  type: {type(d).__name__}, len={len(d)}")
                if isinstance(d, list) and d:
                    log(f"  first item: {d[0] if len(str(d[0])) < 800 else str(d[0])[:800]}")
        except Exception as e:
            log(f"  ERROR: {e}")
    elif pp.suffix == ".csv":
        try:
            text = pp.read_text(encoding="utf-8", errors="ignore")
            file_lines = text.strip().split("\n")
            log(f"  CSV: {len(file_lines)} lines")
            for line in file_lines[:25]:
                log(f"    {line}")
        except Exception as e:
            log(f"  ERROR: {e}")

# Look for pair_manifest files
log("\n=== Search for pair_manifest files ===")
root = Path("I:/Github/Latent_Style/WEAVE")
for item in root.rglob("*"):
    name = item.name.lower()
    if "pair_manifest" in name or "pair-manifest" in name:
        try:
            log(f"  {item} (size={item.stat().st_size if item.is_file() else 'dir'})")
        except Exception:
            log(f"  {item}")

# Look for d5_manifest files
log("\n=== Search for d5_manifest / d5_pair files ===")
for item in root.rglob("*"):
    name = item.name.lower()
    if ("d5" in name and "manifest" in name) or ("d5" in name and "pair" in name):
        try:
            log(f"  {item} (size={item.stat().st_size if item.is_file() else 'dir'})")
        except Exception:
            log(f"  {item}")

# Look for ArtFID evaluator Python script (only top-level tools and utils)
log("\n=== ArtFID evaluator Python scripts (top-level only) ===")
for sub in ["utils", "tools"]:
    sd = root / sub
    if sd.exists():
        for item in sd.rglob("*.py"):
            try:
                text = item.read_text(encoding="utf-8", errors="ignore")
                if "artfid" in text.lower() or "art_fid" in text.lower():
                    log(f"  {item}")
            except Exception:
                pass

# Find baseline output directories under exp/
log("\n=== Baseline output directories under exp/ (top 50) ===")
exp_root = root / "exp"
if exp_root.exists():
    baselines = []
    for item in exp_root.iterdir():
        if item.is_dir():
            name = item.name.lower()
            if any(k in name for k in ["samam", "samst", "style", "zstar", "z-star", "idt", "identity", "seedream", "baseline"]):
                baselines.append(item)
    for b in baselines[:50]:
        try:
            sub_entries = sorted([x.name for x in b.iterdir()])[:8]
            log(f"  {b}")
            for se in sub_entries:
                log(f"      - {se}")
        except Exception:
            log(f"  {b} (cannot list)")

# Look for ArtFID values in existing summary.json files
log("\n=== ArtFID values in summary.json (sample) ===")
count = 0
target_dirs = [
    "exp/repro_weave_d5",
    "exp/compat_baseline_after_cleanup_v2",
    "exp/compat_baseline_after_cleanup",
    "exp/target_style_baseline_p2a",
    "exp/target_style_baseline_r5",
    "exp/710_b0_weave/full_eval/epoch_0010",
    "exp/710_b0_t11/full_eval/epoch_0005",
]
for sub in target_dirs:
    p = root / sub / "summary.json"
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            log(f"\n  {p}")
            # Print all artfid/fid/lpips related keys
            for k, v in d.items():
                kl = k.lower()
                if any(t in kl for t in ["artfid", "art_fid", "fid", "lpips"]):
                    if isinstance(v, (int, float, str)):
                        log(f"    {k} = {v}")
                    elif isinstance(v, dict):
                        log(f"    {k} = dict, keys={list(v.keys())[:10]}")
                    elif isinstance(v, list):
                        log(f"    {k} = list, len={len(v)}")
            count += 1
            if count >= 10:
                break
        except Exception as e:
            log(f"  ERROR {p}: {e}")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Output written to {OUT}")
print(f"Total lines: {len(lines)}")
