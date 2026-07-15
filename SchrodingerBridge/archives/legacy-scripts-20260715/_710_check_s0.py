"""Check S0 training progress."""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

exp_dir = r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_weave"
log_file = os.path.join(exp_dir, "..", "710_b0_weave_log.txt")

# Check checkpoint dir
if os.path.exists(exp_dir):
    print("=== Checkpoint dir ===")
    for f in sorted(os.listdir(exp_dir)):
        full = os.path.join(exp_dir, f)
        if os.path.isfile(full):
            size = os.path.getsize(full)
            print(f"  FILE: {f} ({size} bytes)")
        else:
            print(f"  DIR:  {f}")
else:
    print(f"Checkpoint dir does not exist: {exp_dir}")

# Check log
log_path = r"I:\Github\Latent_Style\SchrodingerBridge\exp\710_b0_weave_log.txt"
if os.path.exists(log_path):
    with open(log_path, "rb") as f:
        raw = f.read()
    if raw[:2] == b"\xff\xfe":
        text = raw.decode("utf-16-le", errors="ignore")
    elif raw[:2] == b"\xfe\xff":
        text = raw.decode("utf-16-be", errors="ignore")
    else:
        text = raw.decode("utf-8", errors="ignore")
    lines = text.split("\n")
    print(f"\n=== Log ({len(lines)} lines) ===")
    print("last 8 non-empty lines:")
    count = 0
    for line in reversed(lines):
        if line.strip():
            print(f"  {line.strip()[-200:]}")
            count += 1
            if count >= 8:
                break
else:
    print(f"Log not found: {log_path}")
