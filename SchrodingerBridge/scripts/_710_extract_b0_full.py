"""Extract B0 complete result line from existing eval+dino data."""
import csv
import json
import os

eval_dir = r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_t11\full_eval\epoch_0005"
metrics_csv = os.path.join(eval_dir, "metrics.csv")
dino_csv = os.path.join(eval_dir, "dino_metrics.csv")
summary_json = os.path.join(eval_dir, "summary.json")
log_file = r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_t11_log.txt"

rows = list(csv.DictReader(open(metrics_csv, encoding="utf-8")))
dino = list(csv.DictReader(open(dino_csv, encoding="utf-8")))
n = len(rows)
off = [i for i, r in enumerate(rows) if r["src_style"] != r["tgt_style"]]
no = len(off)

ac = sum(float(r["clip_style"]) for r in rows) / n
al = sum(float(r["content_lpips"]) for r in rows) / n
ads = sum(float(dino[i]["dino_s"]) for i in range(n)) / n
adc = sum(float(dino[i]["dino_c"]) for i in range(n)) / n
oc = sum(float(rows[i]["clip_style"]) for i in off) / no
ol = sum(float(rows[i]["content_lpips"]) for i in off) / no
ods = sum(float(dino[i]["dino_s"]) for i in off) / no
odc = sum(float(dino[i]["dino_c"]) for i in off) / no

summary = json.load(open(summary_json, encoding="utf-8"))
t = summary.get("timings_sec", {})
wt = t.get("wall_total", 0)
lg = t.get("lancet_generation", 0)

# Parse training time from log (look for total time)
train_min = 0.0
if os.path.exists(log_file):
    with open(log_file, encoding="utf-8", errors="ignore") as f:
        content = f.read()
    # Try to find training time markers
    for line in content.split("\n"):
        if "Total training time" in line or "Training complete" in line:
            # Try to extract minutes
            import re
            m = re.search(r"(\d+\.?\d*)\s*min", line)
            if m:
                train_min = float(m.group(1))
                break

print(f"B0 result line:")
print(f"b0_t11,{train_min:.1f},0,0,{ac:.4f},{al:.4f},{ads:.4f},{adc:.4f},{oc:.4f},{ol:.4f},{ods:.4f},{odc:.4f},{wt:.1f},{lg:.1f}")
