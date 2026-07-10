"""Check D6 training log."""
import os, time
log_path = r"C:\Users\Administrator\logs\d6_style_consist_15ep_train_eval.out"
err_path = r"C:\Users\Administrator\logs\d6_style_consist_15ep_train_eval.err"

for path, label in [(log_path, "STDOUT"), (err_path, "STDERR")]:
    print(f"\n=== {label}: {path} ===")
    if not os.path.exists(path):
        print("  NOT_FOUND")
        continue
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    print(f"  Total lines: {len(lines)}")
    # Print last 30 lines
    for line in lines[-30:]:
        print(f"  {line.rstrip()}")
