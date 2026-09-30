"""Check tails of B0/B1 logs and extract training time."""
import re
import os

logs = [
    r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_t11_log.txt",
    r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b1_no_dwt_route_log.txt",
]
for log in logs:
    if not os.path.exists(log):
        print(f"MISSING: {log}")
        continue
    with open(log, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    print(f"\n=== {os.path.basename(log)} (last 8 lines) ===")
    for line in lines[-8:]:
        print(line.rstrip())
    # Search for training time patterns
    content = "".join(lines)
    # Common patterns: "Total time: 123.4s", "Training took 2.5 min", "epoch_0005", "Saved checkpoint"
    for pattern in [r"[Tt]otal[^\\n]{0,30}(\d+\.?\d*)\s*(min|sec|s)", r"[Tt]rain[^\\n]{0,30}(\d+\.?\d*)\s*(min|sec|s)", r"(\d+\.?\d*)\s*min[^\\n]{0,20}train"]:
        m = re.search(pattern, content)
        if m:
            print(f"  FOUND time: {m.group(0)}")
