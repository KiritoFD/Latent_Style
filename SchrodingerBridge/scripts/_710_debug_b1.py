"""Debug B1 status - check all files."""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

exp_dir = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
exp = "710_b1_no_dwt_route"

# List all files related to b1
print("=== Files in exp dir (b1) ===")
for f in sorted(os.listdir(exp_dir)):
    if "b1" in f:
        full = os.path.join(exp_dir, f)
        if os.path.isfile(full):
            size = os.path.getsize(full)
            print(f"  FILE: {f} ({size} bytes)")
        elif os.path.isdir(full):
            print(f"  DIR:  {f}")

# Check eval dir
eval_dir = os.path.join(exp_dir, exp, "full_eval", "epoch_0005")
print(f"\n=== eval_dir: {eval_dir} ===")
print(f"  exists: {os.path.exists(eval_dir)}")
if os.path.exists(eval_dir):
    for f in sorted(os.listdir(eval_dir)):
        full = os.path.join(eval_dir, f)
        if os.path.isfile(full):
            size = os.path.getsize(full)
            print(f"  FILE: {f} ({size} bytes)")
        elif os.path.isdir(full):
            print(f"  DIR:  {f}")

# Check eval log content (tail)
eval_log = os.path.join(exp_dir, f"{exp}_eval_log.txt")
print(f"\n=== eval_log: {eval_log} ===")
print(f"  exists: {os.path.exists(eval_log)}")
if os.path.exists(eval_log):
    with open(eval_log, "rb") as f:
        raw = f.read()
    print(f"  size: {len(raw)} bytes")
    print(f"  BOM: {raw[:2].hex()}")
    if raw[:2] == b"\xff\xfe":
        text = raw.decode("utf-16-le", errors="ignore")
    elif raw[:2] == b"\xfe\xff":
        text = raw.decode("utf-16-be", errors="ignore")
    else:
        text = raw.decode("utf-8", errors="ignore")
    lines = text.split("\n")
    print(f"  lines: {len(lines)}")
    print("  last 5 non-empty lines:")
    count = 0
    for line in reversed(lines):
        if line.strip():
            print(f"    {line.strip()[-200:]}")
            count += 1
            if count >= 5:
                break

# Check dino log
dino_log = os.path.join(exp_dir, f"{exp}_dino_log.txt")
print(f"\n=== dino_log: {dino_log} ===")
print(f"  exists: {os.path.exists(dino_log)}")
if os.path.exists(dino_log):
    with open(dino_log, "rb") as f:
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
    print("  last 5 non-empty lines:")
    count = 0
    for line in reversed(lines):
        if line.strip():
            print(f"    {line.strip()[-200:]}")
            count += 1
            if count >= 5:
                break
