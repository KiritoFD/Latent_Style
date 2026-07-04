#!/usr/bin/env python3
"""Parse the scan output and produce analysis."""
import json, sys
from pathlib import Path
from collections import defaultdict

INPUT = r"C:\Users\xy\AppData\Local\Temp\trae-agent-toolhost\jobs\job-a87fb28f31e242b9bd09c27853eef557\output.log"

# Read file, skip the progress lines (find the first '{' line)
with open(INPUT, encoding="utf-8") as f:
    lines = f.readlines()

# Find start of JSON (first line that starts with '{')
json_start = 0
for i, line in enumerate(lines):
    if line.strip().startswith("{"):
        json_start = i
        break

json_text = "".join(lines[json_start:])
data = json.loads(json_text)

root_summaries = data["root_summaries"]
experiments = data["experiments"]

print("=" * 80)
print("ROOT SUMMARIES")
print("=" * 80)
for rs in root_summaries:
    print(f"  {rs['root']}: {rs['size']}")

print(f"\nTotal experiments scanned: {len(experiments)}")

# Group by root
by_root = defaultdict(list)
for exp in experiments:
    by_root[exp["root"]].append(exp)

for root, exps in by_root.items():
    print(f"\n  Root '{root}': {len(exps)} experiments")

# Now let's categorize and analyze
def classify_model(name, root):
    """Classify the model type based on directory name and root."""
    n = name.lower()
    if root == "final_works":
        if "cut" in n: return "CUT"
        if "samst" in n: return "SaMST"
        if "star-gan" in n or "stargan" in n: return "StarGAN"
        if "str_0.40" in n: return "SDEdit"
        if "trial" in n: return "ours"
        return "other"
    if "samam" in n: return "SaMam"
    if "samst" in n: return "SaMST"
    if "s2wat" in n: return "S2WAT"
    if "sdedit" in n or "str_0p" in n: return "SDEdit"
    if "sdturbo" in n: return "SDTurbo"
    if "img2img_turbo" in n: return "Img2ImgTurbo"
    if "styleid" in n or "style_id" in n: return "StyleID"
    if "zimage" in n: return "ZImageTurbo"
    if "cyclegan" in n: return "CycleGAN"
    if "cut" in n and "ablate" not in n: return "CUT"
    if "lbm" in n: return "LBM"
    if "flux2" in n: return "Flux2"
    if "aaai2027" in n: return "ours"
    if root == "experiments": return "ours"
    return "other"

def classify_dataset(name, summary_fields):
    """Classify dataset."""
    n = name.lower()
    if "distinct5" in n: return "distinct5"
    if "overfit50" in n: return "overfit50"
    if "wikiart5" in n or "wikiarts5" in n: return "wikiart5"
    if "5x5" in n: return "5x5"
    if "5style" in n: return "5style"
    if "2style" in n: return "2style"
    if "8style" in n: return "8style"
    if "16dim" in n: return "16dim"
    if "256" in n and "legacy" in n: return "legacy256"
    if "512" in n: return "512"
    if "256" in n: return "256"
    return "?"

def format_seconds(sec):
    if sec is None: return "?"
    try:
        sec = float(sec)
        if sec < 60: return f"{sec:.0f}s"
        if sec < 3600: return f"{sec/60:.1f}m"
        return f"{sec/3600:.2f}h"
    except: return str(sec)

# Annotate each experiment
for exp in experiments:
    exp["model_type"] = classify_model(exp["name"], exp["root"])
    exp["dataset"] = classify_dataset(exp["name"], exp.get("summary_fields", {}))
    sf = exp.get("summary_fields", {})
    exp["wall_sec"] = sf.get("wall_seconds") or sf.get("WALL_SECONDS") or sf.get("training_wall_time") or sf.get("train_runtime_sec") or sf.get("runtime_seconds") or sf.get("elapsed_seconds")
    exp["train_steps"] = sf.get("train_steps") or sf.get("total_steps") or sf.get("num_steps") or sf.get("steps") or sf.get("global_step")
    exp["epochs"] = sf.get("epochs") or sf.get("epoch") or sf.get("train_epochs")

# Save annotated data
out_path = Path("g:/GitHub/Latent_Style/SchrodingerBridge/scan_analyzed.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"root_summaries": root_summaries, "experiments": experiments}, f, indent=2, default=str)
print(f"\nAnnotated data saved to: {out_path}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY BY MODEL TYPE")
print("=" * 80)
by_model = defaultdict(list)
for exp in experiments:
    by_model[exp["model_type"]].append(exp)

for model, exps in sorted(by_model.items(), key=lambda x: -len(x[1])):
    total_size_str = exps[0].get("size", "?")
    sizes = [e["size"] for e in exps if e["size"] not in ("?", "TIMEOUT", "ERR")]
    has_summary = sum(1 for e in exps if e["summary_count"] > 0)
    has_ckpt = sum(1 for e in exps if isinstance(e["ckpt_count"], int) and e["ckpt_count"] > 0)
    print(f"  {model:20s}: {len(exps):4d} exps, {has_summary:4d} with summary, {has_ckpt:4d} with ckpt")

# Print Tier 1 - baseline_pipeline/results (the actual baselines)
print("\n" + "=" * 80)
print("TIER 1: Related_Works/baseline_pipeline/results (baselines)")
print("=" * 80)
baseline_exps = [e for e in experiments if e["root"] == "results"]
baseline_exps.sort(key=lambda e: e["mtime"], reverse=True)
print(f"{'name':<60s} {'mtime':<17s} {'size':<8s} {'model':<12s} {'dataset':<10s} {'wall':<8s} {'ckpt':<6s} {'img':<6s}")
for e in baseline_exps:
    wall = format_seconds(e["wall_sec"])
    ckpt = str(e["ckpt_count"])
    img = str(e["img_count"])
    print(f"{e['name'][:60]:<60s} {e['mtime']:<17s} {e['size']:<8s} {e['model_type']:<12s} {e['dataset']:<10s} {wall:<8s} {ckpt:<6s} {img:<6s}")

# Print Tier 2 - exp/ (aaai2027 series)
print("\n" + "=" * 80)
print("TIER 2: exp/ (aaai2027 series + others)")
print("=" * 80)
exp_exps = [e for e in experiments if e["root"] == "exp"]
exp_exps.sort(key=lambda e: e["mtime"], reverse=True)
print(f"{'name':<70s} {'mtime':<17s} {'size':<8s} {'model':<12s} {'wall':<8s} {'ckpt':<6s} {'img':<6s}")
for e in exp_exps:
    wall = format_seconds(e["wall_sec"])
    ckpt = str(e["ckpt_count"])
    img = str(e["img_count"])
    print(f"{e['name'][:70]:<70s} {e['mtime']:<17s} {e['size']:<8s} {e['model_type']:<12s} {wall:<8s} {ckpt:<6s} {img:<6s}")

# Print Tier 3 - experiments/ (historical)
print("\n" + "=" * 80)
print(f"TIER 3: experiments/ (historical ours) - {len(by_root['experiments'])} entries")
print("=" * 80)
hist_exps = [e for e in experiments if e["root"] == "experiments"]
hist_exps.sort(key=lambda e: e["mtime"], reverse=True)
# Print summary stats
sizes = [e["size"] for e in hist_exps if e["size"] not in ("?", "TIMEOUT", "ERR")]
print(f"  Total: {len(hist_exps)} experiments")
print(f"  With summaries: {sum(1 for e in hist_exps if e['summary_count'] > 0)}")
print(f"  With checkpoints: {sum(1 for e in hist_exps if isinstance(e['ckpt_count'], int) and e['ckpt_count'] > 0)}")
print(f"  Date range: {hist_exps[-1]['mtime']} to {hist_exps[0]['mtime']}")
# Print top 30 largest
print("\n  Top 30 largest:")
hist_by_size = sorted(hist_exps, key=lambda e: e["size"], reverse=True)
print(f"  {'name':<55s} {'mtime':<17s} {'size':<8s} {'ckpt':<6s} {'img':<6s}")
for e in hist_by_size[:30]:
    ckpt = str(e["ckpt_count"])
    img = str(e["img_count"])
    print(f"  {e['name'][:55]:<55s} {e['mtime']:<17s} {e['size']:<8s} {ckpt:<6s} {img:<6s}")

# Print final_works
print("\n" + "=" * 80)
print("TIER 4: final_works/")
print("=" * 80)
fw_exps = [e for e in experiments if e["root"] == "final_works"]
fw_exps.sort(key=lambda e: e["mtime"], reverse=True)
print(f"{'name':<30s} {'mtime':<17s} {'size':<8s} {'model':<12s}")
for e in fw_exps:
    print(f"{e['name'][:30]:<30s} {e['mtime']:<17s} {e['size']:<8s} {e['model_type']:<12s}")

# Print Related_Works/runs summary
print("\n" + "=" * 80)
print("Related_Works/runs/")
print("=" * 80)
runs_exps = [e for e in experiments if e["root"] == "runs"]
runs_exps.sort(key=lambda e: e["mtime"], reverse=True)
print(f"{'name':<55s} {'mtime':<17s} {'size':<8s} {'model':<12s}")
for e in runs_exps:
    print(f"{e['name'][:55]:<55s} {e['mtime']:<17s} {e['size']:<8s} {e['model_type']:<12s}")
