"""Extract S0 WEAVE four metrics."""
import csv
import json
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

eval_dir = r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_weave\full_eval\epoch_0010"
rows = list(csv.DictReader(open(eval_dir + r"\metrics.csv", encoding="utf-8")))
n = len(rows)
off = [i for i, r in enumerate(rows) if r["src_style"] != r["tgt_style"]]
no = len(off)

ac = sum(float(r["clip_style"]) for r in rows) / n
al = sum(float(r["content_lpips"]) for r in rows) / n
oc = sum(float(rows[i]["clip_style"]) for i in off) / no
ol = sum(float(rows[i]["content_lpips"]) for i in off) / no

s = json.load(open(eval_dir + r"\dino_summary.json", encoding="utf-8"))

print(f"S0_WEAVE n={n} n_off={no}")
print(f"all: clip_s={ac:.4f} lpips={al:.4f} dino_s={s['all_dino_s']:.4f} dino_c={s['all_dino_c']:.4f} struct={s['all_dino_structure']:.6f}")
print(f"off: clip_s={oc:.4f} lpips={ol:.4f} dino_s={s['off_dino_s']:.4f} dino_c={s['off_dino_c']:.4f} struct={s['off_dino_structure']:.6f}")

# Also extract timing from summary.json
import os
summary_path = eval_dir + r"\summary.json"
if os.path.exists(summary_path):
    sm = json.load(open(summary_path, encoding="utf-8"))
    t = sm.get("timings_sec", {})
    print(f"timing: wall_total={t.get('wall_total',0):.1f}s lancet_gen={t.get('lancet_generation',0):.1f}s")
