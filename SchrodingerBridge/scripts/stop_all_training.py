"""Stop all training processes and report."""
import sys
import subprocess
import time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Kill all python.exe running run.py
print("=== Stopping all training processes ===")
try:
    result = subprocess.run(
        ['wmic', 'process', 'where', "name='python.exe'", 'get', 'processid,commandline'],
        capture_output=True, text=True, timeout=10
    )
    for line in result.stdout.split('\n'):
        if 'run.py' in line:
            parts = line.strip().split()
            pid = parts[-1]
            print(f"Killing PID {pid}: {line.strip()}")
            subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True, timeout=10)
except Exception as e:
    print(f"Error: {e}")

# Also kill any lingering launch scripts
try:
    result = subprocess.run(
        ['wmic', 'process', 'where', "name='python.exe'", 'get', 'processid,commandline'],
        capture_output=True, text=True, timeout=10
    )
    for line in result.stdout.split('\n'):
        if 'launch_latent256' in line or 'wait_check' in line:
            parts = line.strip().split()
            pid = parts[-1]
            print(f"Killing helper PID {pid}: {line.strip()}")
            subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True, timeout=10)
except Exception as e:
    print(f"Error: {e}")

# Delete scheduled task
print("\n=== Deleting scheduled task ===")
subprocess.run(['schtasks', '/delete', '/tn', 'latent256_train', '/f'],
               capture_output=True, text=True, timeout=10)

# Verify GPU freed
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

# Confirm no training python processes
print("=== Remaining python processes ===")
try:
    result = subprocess.run(
        ['wmic', 'process', 'where', "name='python.exe'", 'get', 'processid,commandline'],
        capture_output=True, text=True, timeout=10
    )
    print(result.stdout if result.stdout.strip() else "(none)")
except Exception as e:
    print(f"Error: {e}")
