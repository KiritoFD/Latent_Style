"""Check D6 training status after longer wait."""
import os, time, subprocess

log_path = r"C:\Users\Administrator\logs\d6_style_consist_15ep_train_eval.out"
err_path = r"C:\Users\Administrator\logs\d6_style_consist_15ep_train_eval.err"

# Wait 120 seconds
print("Waiting 120s for training to produce output...")
time.sleep(120)

# Check process
print("\n=== PYTHON PROCESSES ===")
try:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,StartTime,WorkingSet64 | Format-Table -AutoSize"],
        capture_output=True, text=True, timeout=10
    )
    print(result.stdout)
except Exception as e:
    print(f"Error: {e}")

# Check logs
for path, label in [(log_path, "STDOUT"), (err_path, "STDERR")]:
    print(f"\n=== {label} ===")
    if not os.path.exists(path):
        print("  NOT_FOUND")
        continue
    size = os.path.getsize(path)
    print(f"  Size: {size} bytes")
    if size == 0:
        print("  (empty)")
        continue
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    print(f"  Lines: {len(lines)}")
    for line in lines[-50:]:
        print(f"  {line.rstrip()}")

# Check checkpoint dir
ckpt_dir = r"I:\Github\Latent_Style\SchrodingerBridge\exp\d6_style_consist_15ep"
print(f"\n=== CKPT DIR ===")
if os.path.exists(ckpt_dir):
    print(f"  Contents: {os.listdir(ckpt_dir)}")
else:
    print("  NOT_FOUND")
