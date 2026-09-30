"""Check S0 eval progress."""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

eval_dir = r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_weave\full_eval\epoch_0010"
eval_log = r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_weave_eval_log.txt"
dino_log = r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_weave_dino_log.txt"

# Check eval dir
print("=== eval_dir ===")
if os.path.exists(eval_dir):
    for f in sorted(os.listdir(eval_dir)):
        full = os.path.join(eval_dir, f)
        if os.path.isfile(full):
            size = os.path.getsize(full)
            print(f"  FILE: {f} ({size} bytes)")
        else:
            print(f"  DIR:  {f}")
else:
    print("  NOT EXISTS")

# Check eval log
for label, log_path in [("eval_log", eval_log), ("dino_log", dino_log)]:
    print(f"\n=== {label}: {log_path} ===")
    if not os.path.exists(log_path):
        print("  NOT EXISTS")
        continue
    with open(log_path, "rb") as f:
        raw = f.read()
    print(f"  size: {len(raw)} bytes")
    if raw[:2] == b"\xff\xfe":
        text = raw.decode("utf-16-le", errors="ignore")
    elif raw[:2] == b"\xfe\xff":
        text = raw.decode("utf-16-be", errors="ignore")
    else:
        text = raw.decode("utf-8", errors="ignore")
    lines = text.split("\n")
    print(f"  lines: {len(lines)}")
    print("  last 5 non-empty:")
    count = 0
    for line in reversed(lines):
        if line.strip():
            print(f"    {line.strip()[-200:]}")
            count += 1
            if count >= 5:
                break
