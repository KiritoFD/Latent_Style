"""Extract B0 four metrics from metrics.csv and dino_metrics.csv."""
import csv
import math
from pathlib import Path

eval_dir = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_t11\full_eval\epoch_0005")
metrics_csv = eval_dir / "metrics.csv"
dino_csv = eval_dir / "dino_metrics.csv"

rows = list(csv.DictReader(open(metrics_csv, encoding="utf-8")))
dino = list(csv.DictReader(open(dino_csv, encoding="utf-8")))
n = len(rows)
off_idx = [i for i, r in enumerate(rows) if r["src_style"] != r["tgt_style"]]
n_off = len(off_idx)

all_clip = sum(float(r["clip_style"]) for r in rows) / n
all_lpips = sum(float(r["content_lpips"]) for r in rows) / n
all_ds = sum(float(dino[i]["dino_s"]) for i in range(n)) / n
all_dc = sum(float(dino[i]["dino_c"]) for i in range(n)) / n

off_clip = sum(float(rows[i]["clip_style"]) for i in off_idx) / n_off
off_lpips = sum(float(rows[i]["content_lpips"]) for i in off_idx) / n_off
off_ds = sum(float(dino[i]["dino_s"]) for i in off_idx) / n_off
off_dc = sum(float(dino[i]["dino_c"]) for i in off_idx) / n_off

print(f"=== B0 T11 Baseline (n_all={n}, n_off={n_off}) ===")
print(f"all: clip_s={all_clip:.4f} lpips={all_lpips:.4f} dino_s={all_ds:.4f} dino_c={all_dc:.4f}")
print(f"off: clip_s={off_clip:.4f} lpips={off_lpips:.4f} dino_s={off_ds:.4f} dino_c={off_dc:.4f}")

# Also extract timings from summary.json
import json
summary = json.load(open(eval_dir / "summary.json"))
timings = summary.get("timings_sec", {})
print(f"\nTimings:")
print(f"  wall_total: {timings.get('wall_total', 0):.1f}s")
print(f"  lancet_generation: {timings.get('lancet_generation', 0):.1f}s")
print(f"  eval_total: {timings.get('eval_total', 0):.1f}s")
