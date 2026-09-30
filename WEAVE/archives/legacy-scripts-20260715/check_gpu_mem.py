"""Check GPU memory in detail."""
import sys
import subprocess
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Full nvidia-smi with memory info
print("=== nvidia-smi (full) ===")
try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv'],
        capture_output=True, text=True, timeout=10
    )
    print(result.stdout)
except Exception as e:
    print(f"Error: {e}")

# Check log file modification time
import os
import time
log_path = r'C:\Users\Administrator\logs\latent256_train.log'
print(f"\n=== latent256 log file ===")
if os.path.exists(log_path):
    mtime = os.path.getmtime(log_path)
    print(f"Modified: {time.ctime(mtime)}")
    print(f"Size: {os.path.getsize(log_path)} bytes")
else:
    print("NOT FOUND")

# Try to read the first few lines of the log
print("\n=== latent256 log (first 10 lines) ===")
try:
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            print(line, end='')
except Exception as e:
    print(f"Error: {e}")
