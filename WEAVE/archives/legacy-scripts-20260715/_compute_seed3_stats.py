"""Compute 3-seed mean±std for all datasets and metrics."""
import json
import math
import os
import statistics

base = r"I:\Github\Latent_Style\SchrodingerBridge"

seeds = ["seed42", "seed123", "seed2024"]
datasets = ["d5", "p2a", "r5"]

# Collect all metrics
results = {}
for ds in datasets:
    results[ds] = {
        "clip_s_transfer": [], "lpips_transfer": [],
        "clip_s_allpairs": [], "lpips_allpairs": [],
        "dino_c": [], "dino_s": [],
        "gen_time": [], "wall_total": []
    }
    for seed in seeds:
        # Summary
        sp = os.path.join(base, "exp", "seed3", f"{seed}_{ds}_eval", "full_eval", "epoch_0005", "summary.json")
        if not os.path.isfile(sp):
            print(f"MISSING: {sp}")
            continue
        with open(sp) as f:
            s = json.load(f)
        a = s.get("analysis", {})
        st = a.get("style_transfer_ability", {})
        ap = a.get("all_pairs_overview", {})
        t = s.get("timings_sec", {})
        cs_t = st.get("clip_style")
        lp_t = st.get("content_lpips")
        cs_a = ap.get("clip_style")
        lp_a = ap.get("content_lpips")
        gt = t.get("lancet_generation")
        wt = t.get("wall_total")
        print(f"{seed}_{ds}: transfer_clip={cs_t} transfer_lpips={lp_t} allpairs_clip={cs_a} allpairs_lpips={lp_a}")
        results[ds]["clip_s_transfer"].append(cs_t)
        results[ds]["lpips_transfer"].append(lp_t)
        results[ds]["clip_s_allpairs"].append(cs_a)
        results[ds]["lpips_allpairs"].append(lp_a)
        results[ds]["gen_time"].append(gt)
        results[ds]["wall_total"].append(wt)

        # DINO
        dp = os.path.join(base, "exp", "seed3", "_dino", f"{seed}_{ds}.json")
        if not os.path.isfile(dp):
            print(f"MISSING: {dp}")
            continue
        with open(dp) as f:
            d = json.load(f)
        results[ds]["dino_c"].append(d.get("dino_content"))
        results[ds]["dino_s"].append(d.get("dino_style"))

# Compute and print
print("=" * 80)
print("3-Seed Results (mean ± std)")
print("=" * 80)
for ds in datasets:
    print(f"\n--- {ds.upper()} ---")
    for metric in ["clip_s_transfer", "lpips_transfer", "clip_s_allpairs", "lpips_allpairs", "dino_c", "dino_s", "gen_time", "wall_total"]:
        vals = [v for v in results[ds][metric] if v is not None]
        if len(vals) < 2:
            print(f"  {metric:20s}: insufficient data ({len(vals)} vals)")
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {metric:20s}: {mean:.4f} ± {std:.4f}  (vals: {[f'{v:.4f}' for v in vals]})")

# Also compute 1-LPIPS (1 - LPIPS) which is used in the paper
print("\n" + "=" * 80)
print("1-LPIPS (transfer, used in paper)")
print("=" * 80)
for ds in datasets:
    vals = [1 - v for v in results[ds]["lpips_transfer"] if v is not None]
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"  {ds.upper()}: {mean:.4f} ± {std:.4f}  (vals: {[f'{v:.4f}' for v in vals]})")

# Print for paper table (mean±std format) - ALLPAIRS
print("\n" + "=" * 80)
print("Paper Table Format - ALLPAIRS (mean±std)")
print("=" * 80)
for ds in datasets:
    cs = statistics.mean(results[ds]["clip_s_allpairs"])
    cs_std = statistics.stdev(results[ds]["clip_s_allpairs"])
    lp = statistics.mean(results[ds]["lpips_allpairs"])
    lp_std = statistics.stdev(results[ds]["lpips_allpairs"])
    dc = statistics.mean(results[ds]["dino_c"])
    dc_std = statistics.stdev(results[ds]["dino_c"])
    ds_val = statistics.mean(results[ds]["dino_s"])
    ds_std = statistics.stdev(results[ds]["dino_s"])
    print(f"{ds.upper()}: CLIP-S={cs:.4f}±{cs_std:.4f}  LPIPS={lp:.4f}±{lp_std:.4f}  DINO-C={dc:.4f}±{dc_std:.4f}  DINO-S={ds_val:.4f}±{ds_std:.4f}")

# Print for paper table (mean±std format) - TRANSFER
print("\n" + "=" * 80)
print("Paper Table Format - TRANSFER (mean±std)")
print("=" * 80)
for ds in datasets:
    cs = statistics.mean(results[ds]["clip_s_transfer"])
    cs_std = statistics.stdev(results[ds]["clip_s_transfer"])
    lp = statistics.mean(results[ds]["lpips_transfer"])
    lp_std = statistics.stdev(results[ds]["lpips_transfer"])
    dc = statistics.mean(results[ds]["dino_c"])
    dc_std = statistics.stdev(results[ds]["dino_c"])
    ds_val = statistics.mean(results[ds]["dino_s"])
    ds_std = statistics.stdev(results[ds]["dino_s"])
    print(f"{ds.upper()}: CLIP-S={cs:.4f}±{cs_std:.4f}  LPIPS={lp:.4f}±{lp_std:.4f}  DINO-C={dc:.4f}±{dc_std:.4f}  DINO-S={ds_val:.4f}±{ds_std:.4f}")

# Print LaTeX row
print("\n" + "=" * 80)
print("LaTeX Row (ALLPAIRS mean)")
print("=" * 80)
parts = []
for ds in datasets:
    cs = statistics.mean(results[ds]["clip_s_allpairs"])
    lp = statistics.mean(results[ds]["lpips_allpairs"])
    dc = statistics.mean(results[ds]["dino_c"])
    ds_val = statistics.mean(results[ds]["dino_s"])
    parts.extend([f"{cs:.4f}", f"{lp:.4f}", f"{dc:.4f}", f"{ds_val:.4f}"])
print(" & ".join(parts))

# Training speed (from CSV)
print("\n" + "=" * 80)
print("Training Speed (batch=96, 5 epochs)")
print("=" * 80)
csv_path = os.path.join(base, "exp", "seed3", "seed42_b96", "logs")
if os.path.isdir(csv_path):
    csvs = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
    if csvs:
        csv_file = os.path.join(csv_path, csvs[0])
        with open(csv_file) as f:
            lines = f.readlines()
        print(f"CSV: {csvs[0]}")
        for line in lines:
            print(f"  {line.strip()}")

# Inference speed (batch=8, excluding seed42 d5/p2a which used batch=2)
print("\n" + "=" * 80)
print("Inference Speed (batch=8)")
print("=" * 80)
gen_times = []
wall_times = []
for ds in datasets:
    for i, seed in enumerate(seeds):
        gt = results[ds]["gen_time"][i]
        wt = results[ds]["wall_total"][i]
        # Skip seed42 d5 and p2a (old batch=2)
        if seed == "seed42" and ds in ["d5", "p2a"]:
            continue
        gen_times.append(gt)
        wall_times.append(wt)
        print(f"  {seed}_{ds}: gen={gt:.2f}s  wall={wt:.2f}s")

print(f"\n  Gen mean: {statistics.mean(gen_times):.2f}s ± {statistics.stdev(gen_times):.2f}s")
print(f"  Wall mean: {statistics.mean(wall_times):.2f}s ± {statistics.stdev(wall_times):.2f}s")
