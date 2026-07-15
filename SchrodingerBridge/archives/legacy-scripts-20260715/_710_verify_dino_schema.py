"""Verify B6-B8 DINO schema: canonical has dino_structure column, old doesn't."""
import csv
import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

exp_dir = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
exps = [
    "710_b0_t11",
    "710_b1_no_dwt_route",
    "710_b2_det_route",
    "710_b3_p05",
    "710_b4_no_wct",
    "710_b5_strong_ll",
    "710_b6_no_ll",
    "710_b7_2res",
    "710_b8_dim32",
]

for exp in exps:
    dino_csv = os.path.join(exp_dir, exp, "full_eval", "epoch_0005", "dino_metrics.csv")
    if not os.path.exists(dino_csv):
        print(f"{exp}: NO dino_metrics.csv")
        continue
    with open(dino_csv, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
    has_structure = "dino_structure" in header
    print(f"{exp}: header={header} | canonical={has_structure}")
