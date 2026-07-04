"""Stop pixel256, clean up, and report."""
import sys
import subprocess
import os
import shutil
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Kill python processes running pixel256
print("=== Stopping pixel256 training ===")
try:
    result = subprocess.run(
        ['wmic', 'process', 'where', "name='python.exe'", 'get', 'processid,commandline'],
        capture_output=True, text=True, timeout=10
    )
    for line in result.stdout.split('\n'):
        if 'pixel_256' in line:
            parts = line.strip().split()
            pid = parts[-1]
            print(f"Killing PID {pid}: {line.strip()}")
            subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True, timeout=10)
except Exception as e:
    print(f"Error: {e}")

# Delete pixel256 checkpoint dir (only 4% epoch 1, not worth keeping)
pixel256_dir = r'C:\Users\Administrator\exp\pixel256_sfm'
if os.path.exists(pixel256_dir):
    print(f"\nDeleting {pixel256_dir}")
    shutil.rmtree(pixel256_dir, ignore_errors=True)
    print("Deleted")
else:
    print(f"\n{pixel256_dir} not found")

# Verify GPU memory freed
import time
time.sleep(3)
print("\n=== GPU memory after kill ===")
try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv'],
        capture_output=True, text=True, timeout=10
    )
    print(result.stdout)
except Exception as e:
    print(f"Error: {e}")
