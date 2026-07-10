"""Find and kill batch training processes on remote."""
import subprocess
import sys
import os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Find python and powershell processes with command lines
r = subprocess.run(
    ["wmic", "process", "where", "name='python.exe' or name='powershell.exe'",
     "get", "processid,commandline", "/format:csv"],
    capture_output=True, text=True, timeout=15
)
print("=== All python/powershell processes ===")
lines = r.stdout.strip().splitlines()
for line in lines:
    if line.strip():
        print(line[:200])

# Find the batch script process (powershell running _run_t11evo_batch.ps1)
# and the training python process (running run.py)
print("\n=== Looking for batch training processes ===")
r2 = subprocess.run(
    ["wmic", "process", "where", "name='python.exe'",
     "get", "processid,commandline", "/format:csv"],
    capture_output=True, text=True, timeout=15
)
python_pids = []
for line in r2.stdout.strip().splitlines():
    if "run.py" in line.lower() or "t11" in line.lower():
        print(f"TRAINING: {line[:200]}")
        # Extract PID (last field in CSV)
        parts = line.strip().split(",")
        if parts:
            pid = parts[-1].strip()
            if pid.isdigit():
                python_pids.append(pid)

# Find powershell running the batch script
r3 = subprocess.run(
    ["wmic", "process", "where", "name='powershell.exe'",
     "get", "processid,commandline", "/format:csv"],
    capture_output=True, text=True, timeout=15
)
ps_pids = []
for line in r3.stdout.strip().splitlines():
    if "t11evo" in line.lower() or "_run_t11evo" in line.lower():
        print(f"BATCH SCRIPT: {line[:200]}")
        parts = line.strip().split(",")
        if parts:
            pid = parts[-1].strip()
            if pid.isdigit():
                ps_pids.append(pid)

# Kill batch script first (so it doesn't restart training), then kill python
all_pids = ps_pids + python_pids
if all_pids:
    print(f"\n=== Killing PIDs: {all_pids} ===")
    for pid in all_pids:
        os.system(f"taskkill /F /PID {pid}")
        print(f"  Killed PID {pid}")
else:
    print("\nNo training processes found to kill.")

# Verify
import time
time.sleep(2)
r4 = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"],
                     capture_output=True, text=True, timeout=10)
print("\n=== Remaining python processes ===")
print(r4.stdout)
