"""Extract key metrics from Round 2 experiment summaries — fixed field names."""
import json
import os
import csv

REPO = r"I:\Github\Latent_Style\SchrodingerBridge"

EXPERIMENTS = [
    "r2_spec_noswd",
    "r2_spec_llw1",
    "r2_spec_noswd_llw1",
    "r2_spec_swd3",
    "r2_spec_swd6",
    "r2_spectral_10ep",
]

R1_REF = {
    "baseline": (0.7087, 0.2302, 0.3644, 0.0688),
    "spectral": (0.7237, 0.2284, 0.3536, 0.0838),
    "no_swd_loss": (0.7159, 0.2256, 0.3277, 0.0760),
    "ll_w1": (0.7121, 0.2298, 0.3617, 0.0722),
}

print("=== Round 2 Results Summary ===")
print(f"{'Experiment':<25} {'CLIP-S':>8} {'CLIP-T':>8} {'cLPIPS':>8} {'d_idt':>8} {'vs_R1_spec':>12}")
print("-" * 75)

for name in EXPERIMENTS:
    exp_dir = os.path.join(REPO, "exp", name, "full_eval")
    if not os.path.isdir(exp_dir):
        print(f"{name:<25} {'PENDING':>8}")
        continue
    found = False
    for epoch_dir in sorted(os.listdir(exp_dir), reverse=True):
        summary = os.path.join(exp_dir, epoch_dir, "summary.json")
        if not os.path.exists(summary):
            continue
        try:
            with open(summary, encoding="utf-8") as f:
                data = json.load(f)
            apo = data.get("analysis", {}).get("all_pairs_overview", {})
            cs = apo.get("clip_style")
            if cs is not None:
                ct = apo.get("clip_t", 0)
                cl = apo.get("content_lpips", 0)
                di = apo.get("clip_s_delta_idt", 0)
                delta = cs - R1_REF["spectral"][0]
                print(f"{name:<25} {cs:>8.4f} {ct:>8.4f} {cl:>8.4f} {di:>8.4f} {delta:>+12.4f}")
                found = True
                break
        except Exception as e:
            print(f"{name:<25} ERROR: {e}")
            found = True
            break
    if not found:
        print(f"{name:<25} {'PENDING':>8}")

print("\n=== Training Progress (current) ===")
for name in EXPERIMENTS:
    logs_dir = os.path.join(REPO, "exp", name, "logs")
    if not os.path.isdir(logs_dir):
        continue
    csvs = sorted([f for f in os.listdir(logs_dir) if f.startswith("training_") and f.endswith(".csv")], reverse=True)
    if not csvs:
        continue
    csv_path = os.path.join(logs_dir, csvs[0])
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            last = rows[-1]
            num_epochs = len(rows)
            print(f"  {name}: epoch {num_epochs}, loss={last['loss'][:8]}, epoch_time={float(last.get('epoch_time_sec',0)):.0f}s")
    except Exception:
        pass

print("\n=== Runner log ===")
log_path = os.path.join(REPO, "remote_ablation_r2_log.txt")
if os.path.exists(log_path):
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            print(f"  {line.rstrip()}")

print("\n=== Round 1 Reference ===")
for k, v in R1_REF.items():
    print(f"  {k}: CLIP-S={v[0]}, CLIP-T={v[1]}, cLPIPS={v[2]}, d_idt={v[3]}")
