"""Check B2 detailed status."""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

exp_dir = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
exp = "710_b2_det_route"

# Check files
print("=== B2 files ===")
for f in sorted(os.listdir(exp_dir)):
    if "b2" in f:
        full = os.path.join(exp_dir, f)
        if os.path.isfile(full):
            size = os.path.getsize(full)
            print(f"  FILE: {f} ({size} bytes)")
        elif os.path.isdir(full):
            print(f"  DIR:  {f}")

# Check training log tail
log = os.path.join(exp_dir, f"{exp}_log.txt")
if os.path.exists(log):
    with open(log, "rb") as f:
        raw = f.read()
    if raw[:2] == b"\xff\xfe":
        text = raw.decode("utf-16-le", errors="ignore")
    elif raw[:2] == b"\xfe\xff":
        text = raw.decode("utf-16-be", errors="ignore")
    else:
        text = raw.decode("utf-8", errors="ignore")
    lines = text.split("\n")
    print(f"\n=== training log ({len(lines)} lines) ===")
    print("last 5 non-empty lines:")
    count = 0
    for line in reversed(lines):
        if line.strip():
            print(f"  {line.strip()[-200:]}")
            count += 1
            if count >= 5:
                break

# Check eval log
eval_log = os.path.join(exp_dir, f"{exp}_eval_log.txt")
if os.path.exists(eval_log):
    with open(eval_log, "rb") as f:
        raw = f.read()
    if raw[:2] == b"\xff\xfe":
        text = raw.decode("utf-16-le", errors="ignore")
    elif raw[:2] == b"\xfe\xff":
        text = raw.decode("utf-16-be", errors="ignore")
    else:
        text = raw.decode("utf-8", errors="ignore")
    lines = text.split("\n")
    print(f"\n=== eval log ({len(lines)} lines) ===")
    print("last 3 non-empty lines:")
    count = 0
    for line in reversed(lines):
        if line.strip():
            print(f"  {line.strip()[-200:]}")
            count += 1
            if count >= 3:
                break
