"""Check progress of B1-B8 experiments."""
import os
import re
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

exp_dir = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
exps = [
    "710_b1_no_dwt_route",
    "710_b2_det_route",
    "710_b3_p05",
    "710_b4_no_wct",
    "710_b5_strong_ll",
    "710_b6_no_ll",
    "710_b7_2res",
    "710_b8_dim32",
]

# Check results file
results_path = os.path.join(exp_dir, "710_results.txt")
if os.path.exists(results_path):
    with open(results_path, encoding="utf-8") as f:
        lines = f.readlines()
    print(f"=== 710_results.txt ({len(lines)} lines) ===")
    for line in lines:
        print(line.rstrip())

# Check each experiment status
print("\n=== Experiment status ===")
for exp in exps:
    ckpt = os.path.join(exp_dir, exp, "epoch_0005.pt")
    eval_csv = os.path.join(exp_dir, exp, "full_eval", "epoch_0005", "metrics.csv")
    dino_csv = os.path.join(exp_dir, exp, "full_eval", "epoch_0005", "dino_metrics.csv")
    log = os.path.join(exp_dir, f"{exp}_log.txt")
    eval_log = os.path.join(exp_dir, f"{exp}_eval_log.txt")
    dino_log = os.path.join(exp_dir, f"{exp}_dino_log.txt")

    status = []
    if os.path.exists(ckpt):
        status.append("CKPT")
    if os.path.exists(eval_csv):
        status.append("EVAL")
    if os.path.exists(dino_csv):
        status.append("DINO")

    # Check latest log tail
    latest_log = None
    for lg in [dino_log, eval_log, log]:
        if os.path.exists(lg):
            latest_log = lg

    tail = ""
    if latest_log:
        try:
            with open(latest_log, "rb") as f:
                raw = f.read()
            if raw[:2] == b"\xff\xfe":
                text = raw.decode("utf-16-le", errors="ignore")
            elif raw[:2] == b"\xfe\xff":
                text = raw.decode("utf-16-be", errors="ignore")
            else:
                text = raw.decode("utf-8", errors="ignore")
            lines = text.split("\n")
            # Find last non-empty line
            for line in reversed(lines):
                stripped = line.strip()
                if stripped:
                    tail = stripped[-150:]
                    break
        except Exception as e:
            tail = f"ERR: {e}"

    print(f"{exp}: {','.join(status) or 'PENDING'} | tail: {tail}")
